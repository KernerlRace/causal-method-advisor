"""Оценка любого бэкенда на вопросах, написанных вручную (eval/handwritten_questions.jsonl).

  .venv/Scripts/python.exe -m eval.handwritten --backend baseline
  .venv/Scripts/python.exe -m eval.handwritten --backend causalgpt --ckpt checkpoints/sft_pretrained_best.pt --name causalgpt_pretrained

Шаги те же, что в advisor.pipeline.advise: профиль → разбор → диагностики → правила → методы.
Метод считается двумя способами, как в eval/metrics.py: с эталонным разбором (качество одного
ранжирования) и сквозной — с разбором бэкенда; неверные воздействие или исход — ошибка. Диагностики
кэшируются по (файл, разбор): у одного датасета они одинаковые для всех вопросов с тем же разбором.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from advisor.diagnostics import run_diagnostics
from advisor.pipeline import AdviceError, diag_dict, load_backend, query_from_fields, verdict_dict
from advisor.profiler import load_csv, profile
from advisor.rules import evaluate

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "handwritten_questions.jsonl"
RESULTS = ROOT / "eval" / "results"
CONTEXT_KEYS = ("instrument", "mediator", "running", "randomized", "hidden")


def ok(pred: dict, gold: dict, key: str) -> bool:
    g = gold.get(key)
    return pred.get(key) in g if isinstance(g, list) else pred.get(key) == g


def canonical(gold: dict) -> dict:
    """Эталон с одним значением на поле (первая из допустимых альтернатив)."""
    return {k: v[0] if isinstance(v, list) else v for k, v in gold.items()}


class Analyses:
    def __init__(self):
        self.frames, self.cards, self.cache = {}, {}, {}

    def card(self, file: str):
        if file not in self.cards:
            self.frames[file] = load_csv(ROOT / "data" / "demo" / file)
            self.cards[file] = profile(self.frames[file])
        return self.cards[file]

    def run(self, file: str, fields: dict):
        key = (file, json.dumps(fields, sort_keys=True))
        if key not in self.cache:
            df, card = self.frames[file], self.card(file)
            query = query_from_fields(fields, list(df.columns))
            diags = run_diagnostics(df, card, query)
            self.cache[key] = (query, diag_dict(diags), verdict_dict(evaluate(card, query, diags)))
        return self.cache[key]


def rank(backend, an: Analyses, q: dict, fields: dict) -> tuple[dict, str]:
    query, dd, vd = an.run(q["file"], fields)
    card = an.card(q["file"])
    probs, mode = backend.rank(card.structure, dd, vd, fields, question=q["question"], card_text=card.to_text())
    if query.question_type != "not_causal" and not probs:
        mode = "refuse"
    return probs, mode


def evaluate_handwritten(backend, questions: list[dict], explain: bool = False) -> tuple[dict, list[dict]]:
    an = Analyses()
    s = defaultdict(list)
    rows = []
    for q in questions:
        card = an.card(q["file"])
        gold = q["fields"]
        pred = backend.parse(q["question"], card.to_text())
        role_keys = ("target",) if gold["type"] == "not_causal" else ("treatment", "outcome")
        type_ok = pred.get("type") == gold["type"]
        roles_ok = type_ok and all(ok(pred, gold, k) for k in role_keys)
        context_ok = all(ok(pred, gold, k) for k in CONTEXT_KEYS)
        exact = type_ok and roles_ok and context_ok and set(pred) == set(gold)
        s["parse.type"].append(type_ok)
        s["parse.treatment+outcome"].append(roles_ok)
        s["parse.context (iv/mediator/threshold/rct/hidden)"].append(context_ok)
        s["parse.exact"].append(exact)
        s[f"parse.treatment+outcome ({q['lang']})"].append(roles_ok)
        row = {"id": q["id"], "file": q["file"], "lang": q["lang"], "question": q["question"], "gold": gold, "pred": pred}

        if q["best"] is not None:
            probs, mode = rank(backend, an, q, canonical(gold))
            s["top1 (gold parse)"].append(bool(probs) and next(iter(probs)) in q["best"])
            row["methods_gold_parse"] = dict(list(probs.items())[:3])
        e2e_probs, e2e_mode, top1 = {}, None, False
        if roles_ok:
            try:
                e2e_probs, e2e_mode = rank(backend, an, q, pred)
            except AdviceError:
                pass
        if q["best"] is not None:
            top1 = roles_ok and bool(e2e_probs) and next(iter(e2e_probs)) in q["best"]
            s["top1 end-to-end"].append(top1)
            s[f"top1 end-to-end ({q['lang']})"].append(top1)
        if q["mode"] is not None:
            s["mode end-to-end"].append(roles_ok and e2e_mode == q["mode"])
        if gold["type"] == "not_causal":
            s["not-causal questions refused"].append(roles_ok and e2e_mode == "refuse")
        row.update(methods=dict(list(e2e_probs.items())[:3]), mode=e2e_mode, top1_ok=top1)

        if explain and hasattr(backend, "explain") and roles_ok and gold["type"] != "not_causal" and e2e_mode:
            query, dd, vd = an.run(q["file"], pred)
            text = backend.explain(q["question"], card.to_text(), pred, dd, vd, e2e_probs, e2e_mode)
            s["model explanation passes fact check"].append(text is not None)
            row["explanation"] = text
        rows.append(row)
    metrics = {k: f"{np.mean(v):.3f}  (n={len(v)})" for k, v in sorted(s.items())}
    return metrics, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="baseline", choices=["baseline", "causalgpt"])
    ap.add_argument("--ckpt", type=Path, help="чекпойнт для causalgpt")
    ap.add_argument("--name", help="имя в eval/results (по умолчанию — имя бэкенда)")
    ap.add_argument("--explain", action="store_true", help="ещё проверить объяснения модели фактами")
    args = ap.parse_args()

    from threadpoolctl import threadpool_limits
    threadpool_limits(max(1, int((os.cpu_count() or 1) * 0.75)))  # потолок нагрузки на CPU
    kw = {"ckpt": args.ckpt} if args.ckpt else {}
    backend = load_backend(args.backend, **kw)
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    metrics, rows = evaluate_handwritten(backend, questions, explain=args.explain)
    name = args.name or args.backend
    print(f"{name} on {len(questions)} hand-written questions")
    for k, v in metrics.items():
        print(f"  {k:50s} {v}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"handwritten_{name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / f"handwritten_{name}_answers.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> eval/results/handwritten_{name}.json")


if __name__ == "__main__":
    main()
