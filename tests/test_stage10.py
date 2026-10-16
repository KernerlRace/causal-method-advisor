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


def test_role_assumptions_spell_out_instrument_and_mediator():
    from advisor.pipeline import role_assumptions

    assert role_assumptions({"treatment": "t", "outcome": "y"}) == []
    iv = role_assumptions({"treatment": "t", "outcome": "y", "instrument": "z"})
    assert len(iv) == 1 and "`z` is an **instrument**" in iv[0] and "only through `t`" in iv[0]
    fd = role_assumptions({"treatment": "t", "outcome": "y", "mediator": "m"})
    assert len(fd) == 1 and "`m` is a **mediator**" in fd[0]


def test_parsed_mediator_is_stated_and_user_roles_are_not():
    from advisor.pipeline import advise, load_backend
    from advisor.profiler import load_csv

    demos = json.loads((ROOT / "data" / "demo" / "demos.json").read_text(encoding="utf-8"))
    d = next(x for x in demos if x["file"] == "remote_work_frontdoor.csv")
    df, backend = load_csv(ROOT / "data" / "demo" / d["file"]), load_backend("baseline")
    a = advise(df, d["question"], backend)
    assert a.fields.get("mediator") == "skills_score" and any("mediator" in s for s in a.assumptions)
    assert advise(df, d["question"], backend, fields=a.fields).assumptions == []  # роли подтверждены пользователем
