"""Сводная таблица этапа 10: бейзлайн против модели на всех наборах сразу, из уже записанных результатов.

  .venv/Scripts/python.exe -m eval.summary          # -> eval/results/summary.md, summary.json

Ничего не запускает и не обучает: читает eval/results/*.json, которые пишут eval.run_eval, eval.handwritten,
eval.traps и eval.demos. Модель — та, что в приложении (models/causalgpt.pt, запуск sft_overlap); другие
запуски того же рецепта показаны отдельно, чтобы был виден разброс. Для ручных вопросов — парное сравнение
с бейзлайном на одних и тех же вопросах: бутстреп-интервал разности и точный тест Макнемара.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
MODEL = "causalgpt_calibrated"  # ручные вопросы модели из приложения с калиброванным режимом
RUNS = {"causalgpt_calibrated": "shipped run (seed 0, stage 9.3 data)",
        "causalgpt_overlap_seed1": "seed 1, stage 9.3 data",
        "causalgpt_stage9data_seed1": "seed 1, stage 9 data",
        "causalgpt_pretrained": "stage 8 run (seed 0, stage 8 data)"}


def load(name: str):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def value(metrics: dict, key: str) -> float | None:
    """"0.839  (n=4764)" -> 0.839."""
    v = metrics.get(key) if metrics else None
    return None if v is None else float(str(v).split()[0])


def answers(name: str) -> dict[str, dict]:
    path = RESULTS / f"handwritten_{name}_answers.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def paired(a: dict[str, dict], b: dict[str, dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Лучший метод сквозь пайплайн: b против a на вопросах с эталонным методом."""
    ids = [i for i in a if a[i]["best"] is not None]
    x = np.array([a[i]["top1_ok"] for i in ids], float)
    y = np.array([b[i]["top1_ok"] for i in ids], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (n_boot, len(ids)))
    diff = (y[idx] - x[idx]).mean(1)
    only_b, only_a = int(((y == 1) & (x == 0)).sum()), int(((x == 1) & (y == 0)).sum())
    k, n = min(only_a, only_b), only_a + only_b
    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(ids), "baseline": x.mean(), "model": y.mean(), "diff": y.mean() - x.mean(),
            "ci95": (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
            "model_only": only_b, "baseline_only": only_a, "mcnemar_p": p}


def pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def main() -> None:
    rows, out = [], {}

    def row(group: str, metric: str, base: float | None, model: float | None) -> None:
        rows.append(f"| {group} | {metric} | {pct(base)} | {pct(model)} |")
        out.setdefault(group, {})[metric] = {"baseline": base, "model": model}

    bt, mt = load("baseline_test"), load("causalgpt_overlap_test")
    g = "Template test (4,986)"
    row(g, "Best method, given the correct parse", value(bt, "top1 (gold parse)"), value(mt, "top1 (gold parse)"))
    row(g, "Best method, end to end", value(bt, "top1 end-to-end"), value(mt, "top1 end-to-end"))
    row(g, "Treatment and outcome", value(bt, "parse.treatment+outcome"), value(mt, "parse.treatment+outcome"))
    row(g, "Question type, unseen templates", value(bt, "parse.type (heldout tmpl)"), value(mt, "parse.type (heldout tmpl)"))
    row(g, "Answer mode, given the correct parse", value(bt, "mode (gold parse)"), value(mt, "mode (gold parse)"))
    row(g, "Rules only, no ranker (for reference)", value(bt, "top1 rules-only"), None)

    bh, mh = load("handwritten_baseline"), load(f"handwritten_{MODEL}")
    g = "Hand-written (93)"
    row(g, "Best method, end to end", value(bh, "top1 end-to-end"), value(mh, "top1 end-to-end"))
    row(g, "Answer mode, end to end", value(bh, "mode end-to-end"), value(mh, "mode end-to-end"))
    row(g, "Treatment and outcome", value(bh, "parse.treatment+outcome"), value(mh, "parse.treatment+outcome"))
    row(g, "Treatment and outcome, Russian", value(bh, "parse.treatment+outcome (ru)"), value(mh, "parse.treatment+outcome (ru)"))
    row(g, "Question type", value(bh, "parse.type"), value(mh, "parse.type"))
    row(g, "Non-causal questions refused", value(bh, "not-causal questions refused"), value(mh, "not-causal questions refused"))
    row(g, "Model's own explanation passes the fact check", None, value(mh, "model explanation passes fact check"))

    btr, mtr = load("traps_baseline"), load("traps_causalgpt")
    g = "Traps (48)"
    for kind, label in [("all", "All, end to end"), ("unidentifiable", "Hidden confounder, end to end"),
                        ("no_controls", "No control group, end to end"), ("prediction", "Prediction question, end to end"),
                        ("poor_overlap", "Poor overlap, end to end")]:
        row(g, label, btr[kind]["correct"], mtr[kind]["correct"])
    row(g, "All, given the correct parse", btr["all"]["correct_gold_parse"], mtr["all"]["correct_gold_parse"])
    row(g, "All, right or the wrong parse caught", None, mtr["all"].get("correct_or_caught"))

    bd, md = load("demos_baseline"), load("demos_causalgpt")
    if bd and md:
        g = f"Demo questions ({len(bd)})"
        row(g, "As expected (parse, method, mode)", np.mean([r["correct"] for r in bd]), np.mean([r["correct"] for r in md]))

    base = answers("baseline")
    runs = {}
    for name, label in RUNS.items():
        if (RESULTS / f"handwritten_{name}_answers.jsonl").exists():
            runs[name] = {"label": label, **paired(base, answers(name))}
    out["hand-written, paired against the baseline"] = runs

    lines = ["# Baseline against the model", "",
             "Generated by `python -m eval.summary` from `eval/results/`. The model is the one the app ships",
             "(`models/causalgpt.pt`) with the calibrated answer mode.", "",
             "| Set | Metric | Baseline | Model |", "|---|---|---|---|", *rows, "",
             "## Hand-written questions: best method end to end, paired with the baseline", "",
             f"80 questions with a gold method; the baseline gets {pct(next(iter(runs.values()))['baseline'])}. "
             "Interval: paired bootstrap over questions (10,000 resamples); p: exact McNemar test.", "",
             "| Model run | Model | Difference | 95% interval | Only model right / only baseline right | p |",
             "|---|---|---|---|---|---|"]
    for r in runs.values():
        lo, hi = r["ci95"]
        lines.append(f"| {r['label']} | {pct(r['model'])} | {100 * r['diff']:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                     f"| {r['model_only']} / {r['baseline_only']} | {r['mcnemar_p']:.3f} |")
    if bd and md:
        lines += ["", "## Demo questions", "", "| Dataset | Expected | Baseline | Model |", "|---|---|---|---|"]
        for b, m in zip(bd, md):
            def cell(r):
                conf = "" if r.get("confidence") is None else f", {r['confidence']:.2f}"
                ask = ", asks to confirm the " + r["confirm"] if r.get("confirm") else ""
                return f"{'✓' if r['correct'] else '✗'} {r['mode']} `{r['top']}`{conf}{ask}" + (
                    "" if r["parse_ok"] else f" (read {r['treatment']} → {r['outcome']})") + (
                    "" if r["correct"] or not r["parse_ok"] or not r.get("context") else
                    " (read " + ", ".join(f"{k}={v}" for k, v in r["context"].items()) + ")")
            lines.append(f"| `{b['file']}` | {b['expected']} | {cell(b)} | {cell(m)} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
