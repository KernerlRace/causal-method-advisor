"""Итоговая оценка бэкенда на test-выборке (сценарии и шаблоны, которых не было в обучении).

  .venv/Scripts/python.exe -m eval.run_eval --backend baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from advisor.pipeline import BaselineBackend
from eval.metrics import evaluate_backend
from scripts.train_baseline import read_jsonl

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="baseline")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sft", default="data/sft")
    ap.add_argument("--scenarios", default="data/scenarios/scenarios_seed*.jsonl")
    a = ap.parse_args()

    backend = BaselineBackend()
    records = {r["id"]: r for r in read_jsonl(a.scenarios) if "error" not in r}
    examples = [e for e in read_jsonl(f"{a.sft}/{a.split}.jsonl") if e["scenario_id"] in records]
    metrics = evaluate_backend(backend, examples, records)
    print(f"{a.backend} on {a.split}: {len(examples)} questions")
    for k, v in metrics.items():
        print(f"  {k:30s} {v}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{a.backend}_{a.split}.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
