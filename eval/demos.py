"""Демо-сценарии (этап 10): 12 вопросов из data/demo/demos.json сквозь весь пайплайн, как их видит пользователь.

  .venv/Scripts/python.exe -m eval.demos --backend baseline
  .venv/Scripts/python.exe -m eval.demos --backend causalgpt

Это не метрика, а витрина: 12 вопросов ничего не говорят о точности, зато показывают, что видит человек,
который нажимает Advise. Эталон ниже записан до первого запуска этого скрипта и совпадает с ожиданиями из
demos.json и tests/test_stage4.py: допустимые лучшие методы, воздействие и исход. Ответ засчитывается, если
разбор верен, лучший метод из допустимых, а режим — ожидаемый (для отказа — лучший метод не проверяется).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from advisor.pipeline import advise, load_backend
from advisor.profiler import load_csv

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "data" / "demo"
RESULTS = ROOT / "eval" / "results"
CONTEXT_KEYS = ("instrument", "mediator", "running", "randomized", "hidden")

# файл -> (воздействие, исход, допустимые лучшие методы, ожидаемые режимы)
GOLD = {
    "smoking.csv": ("california", "cigsale", {"sdid", "synthetic_control"}, {"recommend", "options"}),
    "marketing_ab_test.csv": ("free_shipping", "conversion_rate", {"diff_in_means", "regression"}, {"recommend", "options"}),
    "training_threshold.csv": ("job_training", "productivity", {"rdd"}, {"recommend"}),
    "tax_instrument.csv": ("tax_rate", "cigsale", {"iv"}, {"recommend"}),
    "drug_rollout_panel.csv": ("newDrug", "recoveryDays", {"sdid", "synthetic_control"}, {"recommend", "options"}),
    "tutoring_staggered.csv": ("Got Tutoring", "Graduated Pct", {"sdid"}, {"recommend", "options"}),
    "push_observational.csv": ("push_enabled", "retained", {"dml", "doubly_robust"}, {"recommend", "options"}),
    "app_metrics_discovery.csv": (None, None, {"lingam", "pc", "ges"}, {"recommend", "options"}),
    "medicaid_timeseries.csv": ("Expansion", "Gdp Per Capita", {"its"}, {"recommend", "options"}),
    "push_unidentifiable.csv": ("got_push", "retention", {"sensitivity_analysis"}, {"refuse"}),
    "remote_work_frontdoor.csv": ("remote_work", "productivity", {"front_door"}, {"recommend"}),
    "physio_continuous.csv": ("physio_hours", "recovery_days", {"regression", "dml"}, {"recommend", "options"}),
}


def run(backend) -> list[dict]:
    rows = []
    for d in json.loads((DEMO_DIR / "demos.json").read_text(encoding="utf-8")):
        treat, out, methods, modes = GOLD[d["file"]]
        a = advise(load_csv(DEMO_DIR / d["file"]), d["question"], backend)
        top = next(iter(a.methods), None)
        parse_ok = treat is None or (a.fields.get("treatment") == treat and a.fields.get("outcome") == out)
        method_ok = a.mode == "refuse" or top in methods
        rows.append({"file": d["file"], "expected": d["expected"],
                     "treatment": a.fields.get("treatment"), "outcome": a.fields.get("outcome"),
                     "context": {k: a.fields[k] for k in CONTEXT_KEYS if a.fields.get(k)},
                     "top": top, "mode": a.mode,
                     "confidence": None if a.confidence is None else round(a.confidence, 3),
                     "parse_confidence": None if a.parse_confidence is None else round(a.parse_confidence, 5),
                     "confirm": a.confirm, "explanation_source": a.explanation_source,
                     "parse_ok": parse_ok, "method_ok": method_ok, "mode_ok": a.mode in modes,
                     "correct": parse_ok and method_ok and a.mode in modes})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="baseline", choices=["baseline", "causalgpt"])
    ap.add_argument("--ckpt", type=Path, help="чекпойнт для causalgpt (по умолчанию models/causalgpt.pt)")
    ap.add_argument("--name", help="имя в eval/results (по умолчанию — имя бэкенда)")
    ap.add_argument("--device", help="cpu / cuda (по умолчанию — cuda, если есть)")
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits
    threadpool_limits(max(1, int((os.cpu_count() or 1) * 0.5)))  # потолок нагрузки на CPU
    kw = {}
    if args.backend == "causalgpt":
        if args.ckpt:
            kw["ckpt"] = args.ckpt
        if args.device:
            kw["device"] = args.device
    backend = load_backend(args.backend, **kw)
    rows = run(backend)
    name = args.name or args.backend
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"demos_{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        mark = "ok " if r["correct"] else "-- "
        conf = "" if r["confidence"] is None else f" conf {r['confidence']:.2f}"
        ask = f" asks to confirm {r['confirm']}" if r["confirm"] else ""
        print(f"{mark}{r['file']:28s} {r['mode']:9s} {str(r['top']):20s} "
              f"T={r['treatment']} Y={r['outcome']}{conf}{ask}")
    print(f"{sum(r['correct'] for r in rows)} of {len(rows)} as expected")


if __name__ == "__main__":
    main()
