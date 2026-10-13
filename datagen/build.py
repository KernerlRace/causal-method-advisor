"""Массовая генерация сценариев: метаданные + профиль + диагностики + вердикты правил в JSONL.

Запуск:
  .venv/Scripts/python.exe -m datagen.build --n 2000 --out data/scenarios --workers 6 --save-csv 100

CSV сохраняются только для первых --save-csv сценариев (для просмотра и оценки): для обучения модели
нужен профиль и диагностики, а не сами таблицы.
"""
from __future__ import annotations

import os

# Параллелим процессами, поэтому внутри процесса — один поток. Иначе каждый воркер запускает
# бустинг/BLAS на все ядра, и потоков становится в разы больше, чем ядер (переподписка тормозит всё).
# Ставится до импорта numpy/sklearn: воркеры на Windows заново импортируют этот модуль.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from advisor.diagnostics import run_diagnostics
from advisor.profiler import profile
from advisor.rules import evaluate
from advisor.diagnostics import all_flags
from datagen.scenario import _jsonable, answer_mode, prune_gold, reconcile_gold
from datagen.scenarios import generate, sample_name


def build_one(args) -> dict:
    index, base_seed, csv_dir = args
    seed = base_seed * 10_000_000 + index
    name = sample_name(np.random.default_rng(seed))
    try:
        s = generate(name, seed)
        card = profile(s.df)
        diags = run_diagnostics(s.df, card, s.query)
        verdicts = evaluate(card, s.query, diags)
        gold_dgp = dict(s.gold)
        if s.answer_mode != "refuse":
            s.gold = prune_gold(reconcile_gold(name, s.gold, all_flags(diags), s.params), len(s.df))
            s.answer_mode = answer_mode(s.gold)
        rec = s.meta()
        rec["gold_dgp"] = gold_dgp  # исходный эталон механизма порождения (для анализа)
        rec.update(
            id=f"{name}-{seed}", seed=seed,
            card_text=card.to_text(),
            structure=card.structure,
            diagnostics={k: {"applicable": r.applicable, "values": r.values, "flags": r.flags,
                             "message": r.message} for k, r in diags.items()},
            verdicts={v.method: {"status": v.status, "reasons": v.reasons, "pros": v.pros, "cons": v.cons}
                      for v in verdicts},
            csv=None,
        )
        if csv_dir is not None:
            path = Path(csv_dir) / f"{rec['id']}.csv"
            s.df.to_csv(path, index=False)
            rec["csv"] = str(path.as_posix())
        return _jsonable(rec)
    except Exception:  # один сломанный сценарий не должен ронять всю сборку
        return {"id": f"{name}-{seed}", "error": traceback.format_exc(limit=3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/scenarios")
    ap.add_argument("--seed", type=int, default=0, help="базовый сид; разные сиды = непересекающиеся наборы")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--save-csv", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    csv_dir = out / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, a.seed, csv_dir if i < a.save_csv else None) for i in range(a.n)]
    t0 = time.perf_counter()
    stats, errors = Counter(), []
    with open(out / f"scenarios_seed{a.seed}.jsonl", "w", encoding="utf-8") as f, \
            ProcessPoolExecutor(a.workers) as pool:
        for i, rec in enumerate(pool.map(build_one, jobs, chunksize=8), 1):
            if "error" in rec:
                errors.append(rec)
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[rec["name"]] += 1
            if i % 500 == 0:
                print(f"  {i}/{a.n}  {i / (time.perf_counter() - t0):.1f} scenarios/s", flush=True)
    dt = time.perf_counter() - t0
    print(f"\n{a.n - len(errors)} scenarios in {dt:.0f}s ({a.n / dt:.1f}/s), {len(errors)} errors")
    for name, c in stats.most_common():
        print(f"  {name:22s} {c}")
    if errors:
        (out / f"errors_seed{a.seed}.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in errors), encoding="utf-8")
        print(f"errors saved, first:\n{errors[0]['error']}")


if __name__ == "__main__":
    main()
