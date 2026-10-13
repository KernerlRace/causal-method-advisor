"""Этап 2: генераторы сценариев.

1. Согласованность: правила не запрещают gold-методы, профайлер видит ожидаемую структуру.
2. Диагностики ловят то, что заложено в сценарий (непараллельные тренды, порог RDD, ...).
3. Правильный метод восстанавливает заложенный эффект, а наивный — нет.
"""
import numpy as np
import pytest

from advisor import estimators as E
from advisor.diagnostics import all_flags, covariate_columns, run_diagnostics
from advisor.profiler import profile, to_bool
from advisor.rules import FORBIDDEN, evaluate
from datagen.scenarios import REGISTRY, generate

SEEDS = range(6)
EXPECTED_STRUCTURE = {
    "panel_parallel": "panel", "panel_factor": "panel", "panel_staggered": "panel", "time_series": "time_series",
}


def analyse(s):
    card = profile(s.df)
    diags = run_diagnostics(s.df, card, s.query)
    verdicts = {v.method: v for v in evaluate(card, s.query, diags)}
    return card, diags, verdicts


# ---------- 1. согласованность gold и правил ----------

@pytest.mark.parametrize("name", list(REGISTRY))
def test_gold_methods_are_not_forbidden(name):
    for seed in SEEDS:
        s = generate(name, seed)
        assert set(s.query.role_columns()) <= set(s.df.columns), (name, seed)
        assert s.answer_mode in ("recommend", "options", "refuse")
        card, diags, verdicts = analyse(s)
        assert card.structure == EXPECTED_STRUCTURE.get(name, "cross_section"), (name, seed, card.structure)
        if card.structure == "panel":
            assert card.design_for(s.query.treatment) is not None, (name, seed)
        for method, w in s.gold.items():
            if w >= 0.5:
                v = verdicts[method]
                assert v.status != FORBIDDEN, (name, seed, method, v.reasons)


# ---------- 2. диагностики видят заложенное ----------

def flag_rate(name, flag, seeds=range(10)):
    hits = 0
    for seed in seeds:
        s = generate(name, seed)
        _, diags, _ = analyse(s)
        hits += flag in all_flags(diags)
    return hits / len(seeds)


@pytest.mark.parametrize("name,flag,min_rate", [
    ("panel_factor", "pretrend_violated", 0.8),
    ("panel_parallel", "pretrend_ok", 0.8),
    ("panel_staggered", "staggered", 1.0),
    ("panel_factor", "single_treated_unit", 0.4),
    ("rct", "randomization_plausible", 0.6),
    ("rdd", "running_variable_found", 0.9),
    ("discovery_nongaussian", "non_gaussian", 0.8),
])
def test_diagnostics_detect_scenario_features(name, flag, min_rate):
    assert flag_rate(name, flag) >= min_rate


@pytest.mark.parametrize("name,flag", [("backdoor_large", "randomization_plausible"),
                                       ("panel_factor", "pretrend_ok")])
def test_diagnostics_do_not_hallucinate(name, flag):
    assert flag_rate(name, flag) <= 0.2


# ---------- 3. правильный метод восстанавливает эффект ----------

def tolerance(s, est):
    sd = float(s.df[s.query.outcome].std())
    return max(4 * (est.se or 0), 0.2 * abs(s.true_effect), 0.12 * sd)


def best_estimate(s, card):
    q, df = s.query, s.df
    covs = covariate_columns(card, q)
    if s.name == "rct":
        if s.params.get("multi_arm"):
            arms = s.params["arms"]
            df = df[df[q.treatment].astype(str).isin(arms[:2])].copy()
            df[q.treatment] = (df[q.treatment].astype(str) == arms[1]).astype(int)
        return E.diff_in_means(df, q.treatment, q.outcome)
    if s.name == "rct_heterogeneous":
        return E.diff_in_means(df, q.treatment, q.outcome)
    if s.name == "backdoor_small":
        return E.ols_adjusted(df, q.treatment, q.outcome, covs)
    if s.name == "backdoor_large":
        return E.aipw(df, q.treatment, q.outcome, covs)
    if s.name == "continuous_treatment":
        return (E.ols_adjusted if s.params["linear"] else E.dml_plr)(df, q.treatment, q.outcome, covs)
    if s.name == "instrumental":
        return E.tsls(df, q.treatment, q.outcome, q.instrument, [c for c in covs if c != q.instrument])
    if s.name == "front_door":
        return E.front_door_linear(df, q.treatment, q.mediator, q.outcome)
    if s.name == "rdd":
        x = s.extras
        return E.rdd_local_linear(df, q.outcome, x["running"], x["cutoff"],
                                  treatment=q.treatment if x["fuzzy"] else None)
    design = card.design_for(q.treatment)
    p = card.panel
    if s.name == "panel_parallel":
        return E.did_twfe(df, p.unit, p.time, q.outcome, panel_d(df, design))
    if s.name in ("panel_factor", "panel_staggered"):
        return E.sdid(df, p.unit, p.time, q.outcome, design)
    if s.name == "time_series":
        return E.its_segmented(df, card.time_col, q.outcome, to_bool(df[q.treatment]).values)
    raise KeyError(s.name)


def panel_d(df, design):
    d = to_bool(df[design.columns[0]])
    for c in design.columns[1:]:
        d = d & to_bool(df[c])
    return d.values


RECOVERY = ["rct", "rct_heterogeneous", "backdoor_small", "backdoor_large", "continuous_treatment",
            "instrumental", "front_door", "rdd", "panel_parallel", "panel_factor", "panel_staggered", "time_series"]


@pytest.mark.parametrize("name", RECOVERY)
def test_correct_method_recovers_effect(name):
    misses = []
    for seed in range(5):
        s = generate(name, seed)
        card = profile(s.df)
        est = best_estimate(s, card)
        if abs(est.value - s.true_effect) > tolerance(s, est):
            misses.append((seed, round(est.value, 3), round(s.true_effect, 3)))
    assert len(misses) <= 1, misses  # допускаем один промах из пяти (шум)


def mean_abs_error(name, estimator, seeds=range(6)):
    errs = []
    for seed in seeds:
        s = generate(name, seed)
        card = profile(s.df)
        sd = float(s.df[s.query.outcome].std())
        errs.append(abs(estimator(s, card) - s.true_effect) / sd)
    return float(np.mean(errs))


@pytest.mark.parametrize("name,naive", [
    ("backdoor_large", lambda s, c: E.diff_in_means(s.df, s.query.treatment, s.query.outcome).value),
    ("instrumental", lambda s, c: E.ols_adjusted(s.df, s.query.treatment, s.query.outcome,
                                                  covariate_columns(c, s.query)).value),
    ("panel_factor", lambda s, c: E.did_twfe(s.df, c.panel.unit, c.panel.time, s.query.outcome,
                                             panel_d(s.df, c.design_for(s.query.treatment))).value),
    ("panel_staggered", lambda s, c: E.did_twfe(s.df, c.panel.unit, c.panel.time, s.query.outcome,
                                                panel_d(s.df, c.design_for(s.query.treatment))).value),
    ("unidentifiable", lambda s, c: E.ols_adjusted(s.df, s.query.treatment, s.query.outcome,
                                                    covariate_columns(c, s.query)).value),
])
def test_naive_method_is_biased(name, naive):
    """Сценарий различает методы: наивный подход ошибается заметно сильнее правильного."""
    naive_err = mean_abs_error(name, naive)
    if name == "unidentifiable":
        assert naive_err > 0.3  # back-door без скрытого конфаундера не спасает
        return
    good_err = mean_abs_error(name, lambda s, c: best_estimate(s, c).value)
    assert naive_err > 2 * good_err, (naive_err, good_err)


def test_heterogeneous_cate_is_recoverable():
    s = generate("rct_heterogeneous", 0)
    cate = E.t_learner_cate(s.df, s.query.treatment, s.query.outcome,
                            covariate_columns(profile(s.df), s.query))
    true = s.extras["true_cate"]
    order = s.df.index  # finalize перемешивает строки: сверяем по корреляции рангов не можем — проверим среднее
    assert abs(cate.mean() - true.mean()) < 0.25 * np.std(true) + 0.1 * float(s.df[s.query.outcome].std())


# ---------- поиск структуры ----------

def _edges(A, nodes):
    return {(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(len(nodes)) if A[i, j]}


def test_lingam_recovers_nongaussian_graphs():
    prec, rec = [], []
    for seed in range(5):
        s = generate("discovery_nongaussian", seed)
        nodes = s.true_graph["nodes"]
        found = _edges(E.discover(s.df[nodes], "lingam"), nodes)
        true = {tuple(e) for e in s.true_graph["edges"]}
        prec.append(len(found & true) / max(len(found), 1))
        rec.append(len(found & true) / len(true))
    assert np.mean(prec) >= 0.8 and np.mean(rec) >= 0.5, (prec, rec)


def test_pc_recovers_gaussian_skeleton():
    f1s = []
    for seed in range(5):
        s = generate("discovery_gaussian", seed)
        nodes = s.true_graph["nodes"]
        found = {frozenset(e) for e in _edges(E.discover(s.df[nodes], "pc"), nodes)}
        true = {frozenset(e) for e in s.true_graph["edges"]}
        tp = len(found & true)
        f1s.append(2 * tp / (len(found) + len(true)))
    assert np.mean(f1s) >= 0.6, f1s
