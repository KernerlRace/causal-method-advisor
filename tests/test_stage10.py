"""Этап 10: демо-сценарии и сводная таблица."""
import json
from pathlib import Path

import pytest

from eval.demos import GOLD
from eval.summary import paired

ROOT = Path(__file__).resolve().parent.parent


def test_every_demo_has_gold():
    demos = json.loads((ROOT / "data" / "demo" / "demos.json").read_text(encoding="utf-8"))
    assert {d["file"] for d in demos} == set(GOLD)
    for treat, out, methods, modes in GOLD.values():
        assert methods and modes <= {"recommend", "options", "refuse"}


def _rows(oks):
    return {f"q{i}": {"best": ["x"], "top1_ok": ok} for i, ok in enumerate(oks)}


def test_paired_comparison():
    base = _rows([1, 1, 0, 0, 0, 0, 1, 0])
    model = _rows([1, 0, 1, 1, 1, 1, 1, 0])
    r = paired(base, model, n_boot=2000)
    assert r["n"] == 8 and r["model_only"] == 4 and r["baseline_only"] == 1
    assert r["diff"] == pytest.approx(3 / 8)
    assert r["mcnemar_p"] == pytest.approx(2 * (1 + 5) / 2 ** 5)  # двусторонний точный тест, 1 из 5 расхождений
    lo, hi = r["ci95"]
    assert lo <= r["diff"] <= hi


def test_paired_skips_questions_without_gold_method():
    base = {**_rows([1, 0]), "refuse": {"best": None, "top1_ok": False}}
    model = {**_rows([1, 1]), "refuse": {"best": None, "top1_ok": False}}
    assert paired(base, model)["n"] == 2


def test_baseline_demos_match_committed_results():
    from advisor.pipeline import load_backend
    from eval.demos import run

    rows = run(load_backend("baseline"))
    saved = json.loads((ROOT / "eval" / "results" / "demos_baseline.json").read_text(encoding="utf-8"))
    assert [(r["file"], r["top"], r["mode"], r["correct"]) for r in rows] == \
           [(r["file"], r["top"], r["mode"], r["correct"]) for r in saved]
