"""Итоговая оценка бэкенда на test-выборке (сценарии и шаблоны, которых не было в обучении).

  .venv/Scripts/python.exe -m eval.run_eval --backend baseline
  .venv/Scripts/python.exe -m eval.run_eval --backend causalgpt --ckpt checkpoints/sft_pretrained_best.pt --name causalgpt_pretrained
  # быстрее: каждый k-й вопрос, всего n
  .venv/Scripts/python.exe -m eval.run_eval --backend causalgpt --ckpt checkpoints/sft_scratch_best.pt --limit 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from advisor.pipeline import load_backend
from eval.metrics import evaluate_backend
from scripts.train_baseline import read_jsonl

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="baseline", choices=["baseline", "causalgpt"])
    ap.add_argument("--ckpt", type=Path, help="чекпойнт для causalgpt")
    ap.add_argument("--name", help="имя файла результата (по умолчанию — имя бэкенда)")
    ap.add_argument("--limit", type=int, help="взять n вопросов, равномерно по файлу")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sft", default="data/sft")
    ap.add_argument("--scenarios", default="data/scenarios/scenarios_seed*.jsonl")
    a = ap.parse_args()

    backend = load_backend(a.backend, **({"ckpt": a.ckpt} if a.ckpt else {}))
    records = {r["id"]: r for r in read_jsonl(a.scenarios) if "error" not in r}
    examples = [e for e in read_jsonl(f"{a.sft}/{a.split}.jsonl") if e["scenario_id"] in records]
    if a.limit and a.limit < len(examples):
        examples = examples[::len(examples) // a.limit][:a.limit]
    metrics = evaluate_backend(backend, examples, records)
    name = a.name or a.backend
    print(f"{name} on {a.split}: {len(examples)} questions")
    for k, v in metrics.items():
        print(f"  {k:30s} {v}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}_{a.split}{'' if not a.limit else f'_{len(examples)}'}.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
