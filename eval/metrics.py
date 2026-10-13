"""Метрики качества для любого бэкенда (бейзлайн сейчас, нейросеть потом).

Проход 2 оценивается двумя способами:
- «given gold parse» — с правильным разбором вопроса (качество одного ранжирования);
- «end-to-end» — с разбором бэкенда. Если воздействие или исход определены неверно, ответ считается
  ошибочным (диагностики посчитались бы не по тем колонкам), это консервативная оценка.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _best(gold: dict) -> set:
    if not gold:
        return set()
    top = max(gold.values())
    return {m for m, w in gold.items() if w == top}


def rules_only_ranking(rec: dict) -> list[str]:
    """Нижняя планка: порядок вердиктов правил (допустимые, затем с оговорками)."""
    return [m for m, v in rec["verdicts"].items() if v["status"] != "forbidden"]


def evaluate_backend(backend, examples: list[dict], records: dict[str, dict]) -> dict:
    from datagen.explanations import parse_fields

    s = defaultdict(list)
    for e in examples:
        rec = records[e["scenario_id"]]
        gold_fields = parse_fields(rec)
        pred = backend.parse(e["question"], e["card"])
        s["parse.type"].append(pred.get("type") == gold_fields["type"])
        roles_ok = all(pred.get(k) == gold_fields.get(k) for k in ("treatment", "outcome", "target"))
        s["parse.treatment+outcome"].append(roles_ok)
        s["parse.exact (all fields)"].append(pred == gold_fields)
        if e["heldout_template"]:
            s["parse.type (heldout tmpl)"].append(pred.get("type") == gold_fields["type"])
        if e["lang"] == "ru":
            s["parse.treatment+outcome (ru)"].append(roles_ok)

        best = _best(rec["gold"])
        probs, mode = backend.rank(rec["structure"], rec["diagnostics"], rec["verdicts"], gold_fields)
        ranked = list(probs)
        s["mode (gold parse)"].append(mode == rec["answer_mode"])
        if best:
            s["top1 (gold parse)"].append(bool(ranked) and ranked[0] in best)
            s["top3 (gold parse)"].append(bool(best & set(ranked[:3])))
            rules = rules_only_ranking(rec)
            s["top1 rules-only"].append(bool(rules) and rules[0] in best)
            ok_parse = pred.get("type") == gold_fields["type"] and roles_ok
            if ok_parse and pred != gold_fields:
                probs, mode = backend.rank(rec["structure"], rec["diagnostics"], rec["verdicts"], pred)
            s["top1 end-to-end"].append(ok_parse and bool(probs) and next(iter(probs)) in best)
        else:  # отказ / не причинный вопрос: верно, если режим refuse
            s["refuse correct (gold parse)"].append(mode == "refuse")
    return {k: f"{np.mean(v):.3f}  (n={len(v)})" for k, v in sorted(s.items())}
