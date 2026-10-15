"""Ловушки (PLAN §8.4): вопросы, на которые честный ответ — отказ или варианты, а не уверенная рекомендация.

  .venv/Scripts/python.exe -m eval.traps --backend baseline
  .venv/Scripts/python.exe -m eval.traps --backend causalgpt

48 вопросов (eval/trap_questions.jsonl) к датасетам data/traps/ (scripts/make_traps.py) и демо, четыре вида:
- unidentifiable — скрытый конфаундер назван, инструмента и медиатора нет: верно только refuse;
- prediction — вопрос о прогнозе, не о причинности: верно только refuse;
- no_controls — все юниты получили воздействие одновременно или ряд слишком короткий: верно options или
  refuse и ни в коем случае не метод, которому нужна контрольная группа (DiD, SC, SDID);
- poor_overlap — воздействие почти полностью определяется ковариатами: верно options или refuse.
Ответ засчитывается, только если тип вопроса и роли (воздействие и исход, для прогноза — целевая колонка)
разобраны верно, как «mode end-to-end» в eval/handwritten.py. Отдельно — ответ при эталонном разборе: он
показывает логику отказов и вариантов отдельно от ошибок разбора. Ожидания записаны до первого запуска
моделей; на этих вопросах ничего не подбирается. Для нейросети считаются оба режима: по калиброванной уверенности
(правило из models/calibration.json) и по токену модели <MODE:*> (как на этапе 8).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from advisor.pipeline import AdviceError, load_backend
from eval.handwritten import Analyses, ok, rank, token_mode

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "trap_questions.jsonl"
RESULTS = ROOT / "eval" / "results"
TRAPS = ("unidentifiable", "no_controls", "prediction", "poor_overlap")
NEEDS_CONTROLS = {"did", "sdid", "synthetic_control"}


def answer(backend, an: Analyses, q: dict, fields: dict) -> dict:
    """Режим ответа (и по токену модели), уверенность и лучший метод при данном разборе."""
    probs, mode, token, conf = {}, "error", None, None
    try:
        probs, mode = rank(backend, an, q, fields)
        token = token_mode(backend, mode)
        last = getattr(backend, "last", None)
        conf = last.confidence if last is not None and getattr(backend, "calibrated", False) else None
    except AdviceError:
        pass  # разбор, по которому нельзя построить вопрос (нет воздействия и т. п.)
    top = next(iter(probs), None)
    safe = top not in NEEDS_CONTROLS
    return {"mode": mode, "token_mode": token, "confidence": conf, "top": top,
            "methods": dict(list(probs.items())[:3]),
            "ok": mode in q["expect"] and safe, "ok_token": token in q["expect"] and safe}


def evaluate_traps(backend, questions: list[dict]) -> tuple[dict, list[dict]]:
    an = Analyses(ROOT / "data")
    rows = []
    for q in questions:
        card = an.card(q["file"])
        gold = q["fields"]
        pred = backend.parse(q["question"], card.to_text())
        role_keys = ("target",) if gold["type"] == "not_causal" else ("treatment", "outcome")
        parse_ok = pred.get("type") == gold["type"] and all(ok(pred, gold, k) for k in role_keys)
        e2e, given = answer(backend, an, q, pred), answer(backend, an, q, gold)
        rows.append({"id": q["id"], "trap": q["trap"], "file": q["file"], "lang": q["lang"],
                     "question": q["question"], "gold": gold, "pred": pred, "parse_ok": parse_ok,
                     **{k: e2e[k] for k in ("mode", "token_mode", "confidence", "top", "methods")},
                     "expect": q["expect"],
                     "correct": parse_ok and e2e["ok"], "correct_token": parse_ok and e2e["ok_token"],
                     "gold_parse": {k: given[k] for k in ("mode", "token_mode", "confidence", "top")},
                     "correct_gold_parse": given["ok"], "correct_gold_parse_token": given["ok_token"]})
    return summarize(rows, token=hasattr(backend, "last")), rows


def summarize(rows: list[dict], token: bool) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["trap"]].append(r)
        by["all"].append(r)
    out = {}
    for trap in (*TRAPS, "all"):
        rs = by[trap]
        if not rs:
            continue
        n = len(rs)
        modes = Counter(r["mode"] for r in rs)
        out[trap] = {
            "n": n,
            "correct": round(sum(r["correct"] for r in rs) / n, 3),
            "parse_ok": round(sum(r["parse_ok"] for r in rs) / n, 3),
            "modes": {m: modes.get(m, 0) for m in ("refuse", "options", "recommend", "error")},
            "needs_controls_recommended": sum(r["top"] in NEEDS_CONTROLS and r["mode"] != "refuse" for r in rs),
            "correct_gold_parse": round(sum(r["correct_gold_parse"] for r in rs) / n, 3),
        }
        if token:
            tm = Counter(r["token_mode"] for r in rs)
            out[trap]["correct_token"] = round(sum(r["correct_token"] for r in rs) / n, 3)
            out[trap]["modes_token"] = {m: tm.get(m, 0) for m in ("refuse", "options", "recommend")}
            out[trap]["correct_gold_parse_token"] = round(sum(r["correct_gold_parse_token"] for r in rs) / n, 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="baseline", choices=["baseline", "causalgpt"])
    ap.add_argument("--ckpt", type=Path, help="чекпойнт для causalgpt (калибровка — только для своей модели)")
    ap.add_argument("--name", help="имя в eval/results (по умолчанию — имя бэкенда)")
    ap.add_argument("--device", help="cpu / cuda (по умолчанию — cuda, если есть)")
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits
    threadpool_limits(max(1, int((os.cpu_count() or 1) * 0.5)))  # потолок нагрузки на CPU
    kw = {k: v for k, v in (("ckpt", args.ckpt), ("device", args.device)) if v}
    backend = load_backend(args.backend, **kw)
    if args.backend == "causalgpt":
        from model.limits import GpuDuty
        backend.model.pause = GpuDuty(backend.model.device, duty=0.5).step
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    metrics, rows = evaluate_traps(backend, questions)
    name = args.name or args.backend
    print(f"{name} on {len(questions)} trap questions (correct = expected mode with the right parse)")
    for trap, m in metrics.items():
        line = (f"  {trap:15s} n={m['n']:2d}  correct {m['correct']:.0%}  parse {m['parse_ok']:.0%}  modes {m['modes']}"
                f"  | given the gold parse: {m['correct_gold_parse']:.0%}")
        if "correct_token" in m:
            line += (f"\n  {'':15s} by the mode token: correct {m['correct_token']:.0%}  modes {m['modes_token']}"
                     f"  | given the gold parse: {m['correct_gold_parse_token']:.0%}")
        print(line)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"traps_{name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    with (RESULTS / f"traps_{name}_answers.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> eval/results/traps_{name}.json")


if __name__ == "__main__":
    main()
