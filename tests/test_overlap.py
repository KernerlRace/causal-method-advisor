"""Слабое пересечение групп в обучающих данных: back-door с почти предрешённым воздействием → варианты.

1. Сценарий backdoor_targeted действительно даёт poor_overlap и не похож на порог RDD.
2. Эталон при poor_overlap — варианты при любом размере данных; RDD остаётся RDD.
3. Новый сценарий не сдвигает старые: остальные сиды дают прежние имена.
4. Объяснение называет проблему и берёт числа только из диагностик.
"""
import re

import numpy as np
import pytest

from advisor.diagnostics import all_flags, run_diagnostics
from advisor.profiler import profile
from datagen.build import build_one
from datagen.explanations import target
from datagen.scenario import answer_mode, backdoor_gold, overlap_gold, prune_gold, reconcile_gold
from datagen.scenarios import ADDED, REGISTRY, generate, sample_name


def flags_of(name, seed):
    s = generate(name, seed)
    card = profile(s.df)
    return s, all_flags(run_diagnostics(s.df, card, s.query))


# ---------- 1. сценарий ----------

def test_targeted_scenario_has_poor_overlap_and_no_threshold():
    rows = [flags_of("backdoor_targeted", seed) for seed in range(8)]
    assert sum("poor_overlap" in f for _, f in rows) >= 6
    assert not any("running_variable_found" in f for _, f in rows)  # правило по нескольким ковариатам, не порог
    for s, f in rows:
        g = prune_gold(reconcile_gold(s.name, s.gold, f, s.params), len(s.df))
        if "poor_overlap" in f:
            assert answer_mode(g) == "options" and max(g, key=g.get) == "doubly_robust"


# ---------- 2. эталон ----------

@pytest.mark.parametrize("n", [60, 150, 500, 20000])
@pytest.mark.parametrize("k", [3, 6])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_overlap_gold_is_options_at_any_size(n, k, nonlinear):
    g = overlap_gold(n, k, nonlinear)
    assert answer_mode(g) == "options"
    assert max(g.values()) == 1.0
    assert g.get("ipw", 0) < 0.5  # экстремальные веса
    if n >= 200:
        assert {m for m, w in g.items() if w >= 0.9} == {"doubly_robust", "psm"}
        assert g["regression"] < 0.9  # молча экстраполирует


def test_reconcile_poor_overlap_by_scenario():
    small = reconcile_gold("backdoor_small", backdoor_gold(300, 3), {"poor_overlap", "linear_ok"}, {"n": 300, "k": 3})
    assert answer_mode(small) == "options"
    large = reconcile_gold("backdoor_large", backdoor_gold(9000, 10), {"poor_overlap", "nonlinear"},
                           {"n": 9000, "k": 10})
    assert answer_mode(large) == "options" and "dml" not in large
    # без флага прежние ответы: регрессия на малых данных, DML и DR на больших
    assert reconcile_gold("backdoor_small", backdoor_gold(300, 3), {"good_overlap"}, {"n": 300, "k": 3}) == \
        backdoor_gold(300, 3)
    soft = reconcile_gold("backdoor_targeted", overlap_gold(400, 3, False), {"linear_ok"}, {"n": 400, "k": 3})
    assert answer_mode(soft) == "recommend" and max(soft, key=soft.get) == "regression"
    soft = reconcile_gold("backdoor_targeted", overlap_gold(9000, 3, False), {"linear_ok"}, {"n": 9000, "k": 3})
    assert soft == {**backdoor_gold(9000, 3), "regression": 0.6}
    # в RDD пересечения нет по построению, эффект оценивается у порога
    assert reconcile_gold("rdd", {"rdd": 1.0}, {"poor_overlap", "running_variable_found"}, {"n": 900}) == {"rdd": 1.0}


# ---------- 3. реестр ----------

def test_added_scenario_keeps_earlier_draws():
    base = [n for n in REGISTRY if n not in ADDED]
    w = np.array([REGISTRY[n][1] for n in base])
    seeds = range(10_000_000, 10_006_000)
    names = [sample_name(np.random.default_rng(s)) for s in seeds]
    for s, name in zip(seeds, names):
        if name not in ADDED:
            assert name == base[int(np.random.default_rng(s).choice(len(base), p=w / w.sum()))]
    share = np.mean([n == "backdoor_targeted" for n in names])
    expected = REGISTRY["backdoor_targeted"][1] / sum(v[1] for v in REGISTRY.values())
    assert abs(share - expected) < 0.01


# ---------- 4. объяснение ----------

NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w%])")


def test_overlap_explanation_names_the_problem():
    index = next(i for i in range(200) if sample_name(np.random.default_rng(10_000_000 + i)) == "backdoor_targeted")
    rec = build_one((index, 1, None))
    assert rec["answer_mode"] == "options", rec.get("error")
    for lang, word in (("en", "overlap"), ("ru", "пересека")):
        t = target(rec, lang, np.random.default_rng(0))
        text = t["explanation"]
        assert word in text and "propensity" in text
        assert "Options:" in text or "Варианты:" in text
        src = rec["card_text"] + t["diag"]
        assert set(NUM.findall(text)) <= set(NUM.findall(src)), text
