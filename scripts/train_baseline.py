"""Обучение бейзлайна (этап 4) и быстрая проверка на val.

  .venv/Scripts/python.exe scripts/train_baseline.py --max-parser 40000
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402

from advisor.baseline import MethodRanker, QuestionParser  # noqa: E402
from advisor.pipeline import MODELS_DIR  # noqa: E402
from datagen.build_sft import split_of  # noqa: E402
from eval.metrics import evaluate_backend  # noqa: E402


def read_jsonl(pattern: str, limit: int | None = None) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # файл может дописываться прямо сейчас
                    continue
                if limit and len(out) >= limit:
                    return out
    return out


class _Bundle:
    name = "baseline"

    def __init__(self, parser, ranker):
        self.parser, self.ranker = parser, ranker

    def parse(self, q, card_text):
        return self.parser.predict(q, card_text)

    def rank(self, *a, **context):  # вопрос и профиль (question=, card_text=) ранжировщику бейзлайна не нужны
        return self.ranker.predict(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/sft")
    ap.add_argument("--scenarios", default="data/scenarios/scenarios_seed*.jsonl")
    ap.add_argument("--max-parser", type=int, default=40000)
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "baseline.joblib")
    a = ap.parse_args()

    t0 = time.perf_counter()
    train_ex = read_jsonl(f"{a.sft}/train.jsonl", a.max_parser)
    records = [r for r in read_jsonl(a.scenarios) if "error" not in r]
    train_rec = [r for r in records if split_of(r["id"]) == "train"]
    print(f"parser on {len(train_ex)} questions, ranker on {len(train_rec)} scenarios")
    parser = QuestionParser().fit(train_ex)
    ranker = MethodRanker().fit(train_rec)
    a.out.parent.mkdir(exist_ok=True)
    joblib.dump({"parser": parser, "ranker": ranker}, a.out, compress=3)
    print(f"trained in {time.perf_counter() - t0:.0f}s -> {a.out}")

    by_id = {r["id"]: r for r in records}
    val_ex = [e for e in read_jsonl(f"{a.sft}/val.jsonl") if e["scenario_id"] in by_id][:3000]
    for k, v in evaluate_backend(_Bundle(parser, ranker), val_ex, by_id).items():
        print(f"  val {k:28s} {v}")


if __name__ == "__main__":
    main()
