"""Сборка обучающего набора: сценарии (datagen.build) → вопросы + целевые ответы → train/val/test.

  .venv/Scripts/python.exe -m datagen.build_sft --scenarios "data/scenarios/scenarios_seed*.jsonl" --per-scenario 2

Разбиение по сценариям (один датасет не попадает в разные части). Отложенные шаблоны вопросов
встречаются только в test: так видно, понимает ли модель незнакомые формулировки.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from datagen.explanations import target
from datagen.questions import is_heldout, make_question

SPLITS = (("test", 0.05), ("val", 0.05), ("train", 0.90))


def split_of(scenario_id: str) -> str:
    h = int(hashlib.md5(scenario_id.encode()).hexdigest(), 16) % 1000 / 1000
    acc = 0.0
    for name, share in SPLITS:
        acc += share
        if h < acc:
            return name
    return "train"


def examples_for(rec: dict, per_scenario: int) -> list[dict]:
    split = split_of(rec["id"])
    out = []
    for k in range(per_scenario):
        rng = np.random.default_rng([rec["seed"], k, 7])
        q = make_question(rec, rng, allow_heldout=(split == "test"))
        t = target(rec, q["lang"], rng)
        out.append({
            "id": f"{rec['id']}#q{k}", "scenario_id": rec["id"], "split": split, "name": rec["name"],
            "lang": q["lang"], "template_id": q["template_id"], "heldout_template": is_heldout(q["template_id"]),
            "card": rec["card_text"], "question": q["text"], **t,
        })
    return out


def review_markdown(examples: list[dict], rng, n: int = 50) -> str:
    """Выборка для ручной проверки: по возможности все сценарии и оба языка."""
    by_name: dict[str, list] = {}
    for e in examples:
        by_name.setdefault(e["name"], []).append(e)
    picked = []
    while len(picked) < n and any(by_name.values()):
        for name in sorted(by_name):
            if by_name[name] and len(picked) < n:
                picked.append(by_name[name].pop(int(rng.integers(len(by_name[name])))))
    parts = ["# SFT review sample\n"]
    for i, e in enumerate(picked, 1):
        parts.append(f"## {i}. {e['name']} ({e['lang']}, mode={e['mode']})\n")
        parts.append(f"**Question:** {e['question']}\n")
        parts.append(f"```\n{e['card']}\n```\n")
        parts.append(f"**Parse:** `{e['parse']}`\n")
        parts.append(f"```\n{e['diag']}\n```\n")
        parts.append(f"**Methods:** {e['methods']}\n")
        parts.append(f"**Explanation:**\n\n{e['explanation']}\n")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="data/scenarios/scenarios_seed*.jsonl")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--review", default="docs/sft_review_sample.md")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {name: open(out / f"{name}.jsonl", "w", encoding="utf-8") for name, _ in SPLITS}
    stats, lengths, sample_pool = Counter(), [], []
    for path in sorted(glob.glob(a.scenarios)):
        for line in open(path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # файл сценариев может ещё дописываться
                continue
            for ex in examples_for(rec, a.per_scenario):
                files[ex["split"]].write(json.dumps(ex, ensure_ascii=False) + "\n")
                stats[(ex["split"], "all")] += 1
                stats[(ex["split"], ex["lang"])] += 1
                stats[("mode", ex["mode"])] += 1
                stats[(ex["split"], "heldout")] += ex["heldout_template"]
                lengths.append(len(ex["card"]) + len(ex["question"]) + len(ex["parse"]) + len(ex["diag"])
                               + len(ex["explanation"]))
                if len(sample_pool) < 5000:
                    sample_pool.append(ex)
    for f in files.values():
        f.close()
    for name, _ in SPLITS:
        n = stats[(name, "all")]
        print(f"{name:5s} {n:7d} examples  en={stats[(name, 'en')]} ru={stats[(name, 'ru')]} "
              f"heldout-template={stats[(name, 'heldout')]}")
    print("modes:", {k[1]: v for k, v in stats.items() if k[0] == "mode"})
    L = np.array(lengths)
    print(f"chars per example: median {np.median(L):.0f}, p95 {np.percentile(L, 95):.0f}, max {L.max()}")
    if a.review:
        Path(a.review).parent.mkdir(parents=True, exist_ok=True)
        Path(a.review).write_text(review_markdown(sample_pool, np.random.default_rng(0)), encoding="utf-8")
        print(f"review sample -> {a.review}")


if __name__ == "__main__":
    main()
