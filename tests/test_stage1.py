"""Этап 1: профайлер, диагностики и правила на демо-данных и синтетике."""
import numpy as np
import pytest

from advisor.diagnostics import all_flags, run_diagnostics
from advisor.knowledge import CardError, load_cards, validate
from advisor.profiler import profile
from advisor.query import CausalQuery
from advisor.rules import ALLOWED, FORBIDDEN, WARNING, evaluate


def verdicts(df, query):
    card = profile(df)
    diags = run_diagnostics(df, card, query)
    return {v.method: v for v in evaluate(card, query, diags)}, all_flags(diags), card


def usable(vs):
    return {m for m, v in vs.items() if v.status != FORBIDDEN}


# ---------- карточки ----------

def test_cards_load_and_are_valid():
    cards = load_cards()
    assert len(cards) == 22
    assert {"did", "synthetic_control", "sdid", "iv", "rdd", "pc", "lingam"} <= set(cards)


def test_card_validation_catches_unknown_flag():
    raw = {"id": "x", "name": "x", "name_ru": "x", "family": "rct", "question_types": ["effect"],
           "estimand": "ATE", "summary": "x", "requires": {"structure": ["panel"], "treatment_types": ["binary"]},
           "assumptions": [], "diagnostics": [], "libraries": [], "sources": [],
           "good_flags": {"no_such_flag": "x"}}
    with pytest.raises(CardError):
        validate(raw, "x.yaml")


# ---------- критерий этапа 1: Prop 99 ----------

def test_smoking_is_panel_with_single_treated_state(smoking):
    card = profile(smoking)
    assert card.structure == "panel"
    assert (card.panel.unit, card.panel.time) == ("state", "year")
    d = card.design_for("california")
    assert d.columns == ("california", "after_treatment")
    assert (d.n_treated_units, d.n_control_units, d.n_pre_periods, d.n_post_periods) == (1, 38, 19, 12)
    assert not d.staggered


def test_smoking_rejects_parallel_trends_and_prefers_synthetic_methods(smoking):
    vs, flags, _ = verdicts(smoking, CausalQuery("effect", treatment="california", outcome="cigsale"))
    assert "pretrend_violated" in flags
    assert vs["synthetic_control"].status == ALLOWED, (sorted(flags), vs["synthetic_control"].cons)
    assert vs["sdid"].status == ALLOWED, (sorted(flags), vs["sdid"].cons)
    assert vs["did"].status == WARNING
    assert any("not parallel" in c for c in vs["did"].cons)
    assert vs["psm"].status == FORBIDDEN  # кросс-секционный метод на панели


# ---------- другие структуры ----------

def test_rct_balance_looks_random(rct):
    vs, flags, card = verdicts(rct, CausalQuery("effect", treatment="got_coupon", outcome="spend", randomized=True))
    assert card.structure == "cross_section"
    assert "randomization_plausible" in flags and "imbalance" not in flags
    assert vs["diff_in_means"].status == ALLOWED
    assert vs["did"].status == FORBIDDEN


def test_rct_unconfirmed_randomization_gives_warning(rct):
    vs, _, _ = verdicts(rct, CausalQuery("effect", treatment="got_coupon", outcome="spend"))
    assert vs["diff_in_means"].status == WARNING


def test_observational_flags_poor_overlap(observational):
    vs, flags, _ = verdicts(observational, CausalQuery("effect", treatment="treatment", outcome="outcome",
                                                       randomized=False))
    assert {"poor_overlap", "imbalance"} <= flags
    assert "randomization_plausible" not in flags
    assert vs["diff_in_means"].status == FORBIDDEN
    assert vs["ipw"].status == WARNING
    assert vs["dml"].status != FORBIDDEN


def test_rdd_running_variable_is_detected(rdd):
    card = profile(rdd)
    q = CausalQuery("effect", treatment="scholarship", outcome="earnings")
    diags = run_diagnostics(rdd, card, q)
    rv = diags["running_variable"]
    assert rv.values["variable"] == "test_score"
    assert abs(rv.values["cutoff"] - 60) < 2
    assert {"running_variable_found", "sharp_cutoff"} <= set(rv.flags)
    vs = {v.method: v for v in evaluate(card, q, diags)}
    assert vs["rdd"].status == ALLOWED


def test_no_running_variable_in_observational(observational):
    vs, flags, _ = verdicts(observational, CausalQuery("effect", treatment="treatment", outcome="outcome"))
    assert "running_variable_found" not in flags
    assert vs["rdd"].status == FORBIDDEN


def test_iv_needs_instrument_and_reports_first_stage(iv):
    q = CausalQuery("effect", treatment="education_years", outcome="wage")
    vs, _, _ = verdicts(iv, q)
    assert vs["iv"].status == FORBIDDEN
    q = CausalQuery("effect", treatment="education_years", outcome="wage", instrument="distance_low",
                    hidden_confounding=True)
    vs, flags, _ = verdicts(iv, q)
    assert vs["iv"].status in (ALLOWED, WARNING)
    assert "weak_instrument" not in flags
    assert vs["iv"].pros  # hidden_confounding -> аргумент за IV
    assert vs["psm"].status == FORBIDDEN  # непрерывное воздействие


def test_time_series_structure(time_series):
    card = profile(time_series)
    assert card.structure == "time_series" and card.time_col == "date"
    vs, _, _ = verdicts(time_series, CausalQuery("effect", treatment="campaign_live", outcome="sales"))
    assert vs["its"].status != FORBIDDEN
    assert vs["did"].status == FORBIDDEN


def test_staggered_panel(staggered_panel):
    card = profile(staggered_panel)
    assert card.structure == "panel" and card.panel.unit == "store_id"
    d = card.design_for("new_layout")
    assert d.kind == "indicator" and d.staggered
    assert set(d.cohorts) == {2015, 2018} and d.n_treated_units == 5
    vs, flags, _ = verdicts(staggered_panel, CausalQuery("effect", treatment="new_layout", outcome="revenue"))
    assert "staggered" in flags
    assert any("staggered" in c or "different times" in c for c in vs["did"].cons)
    assert any("cohort" in p for p in vs["sdid"].pros)


def test_parallel_panel_keeps_did(parallel_panel):
    vs, flags, _ = verdicts(parallel_panel, CausalQuery("effect", treatment="treated_group", outcome="y"))
    assert "pretrend_ok" in flags
    assert vs["did"].status == ALLOWED
    assert any("parallel" in p for p in vs["did"].pros)
    # колонка «после» уже в паре с группой — отдельного дизайна «все получили воздействие» нет
    assert [(d.kind, d.columns) for d in profile(parallel_panel).designs] == [("group_post", ("treated_group", "post"))]


def test_categorical_outcome_does_not_crash_diagnostics(observational):
    """Разбор вопроса может назвать исходом категориальную колонку: проверка нелинейности тогда неприменима."""
    vs, flags, _ = verdicts(observational, CausalQuery("effect", treatment="treatment", outcome="region"))
    card = profile(observational)
    nl = run_diagnostics(observational, card, CausalQuery("effect", treatment="treatment", outcome="region"))["nonlinearity"]
    assert not nl.applicable and "not numeric" in nl.message
    yes_no = observational.assign(good=lambda d: (d["outcome"] > 0).map({True: "yes", False: "no"}))
    nl = run_diagnostics(yes_no, profile(yes_no), CausalQuery("effect", treatment="treatment", outcome="good"))
    assert nl["nonlinearity"].applicable


def test_outcome_with_missing_values_does_not_crash_diagnostics(observational):
    """Исходом может оказаться колонка с пропусками (так прочитан вопрос или такой CSV): строки без исхода
    в проверке нелинейности не участвуют, а если известных почти нет — проверка неприменима."""
    df = observational.copy()
    df.loc[df.index[::4], "outcome"] = np.nan
    q = CausalQuery("effect", treatment="treatment", outcome="outcome")
    assert run_diagnostics(df, profile(df), q)["nonlinearity"].applicable
    df.loc[df.index[20:], "outcome"] = np.nan
    nl = run_diagnostics(df, profile(df), q)["nonlinearity"]
    assert not nl.applicable and "known" in nl.message


def test_panel_where_every_unit_is_treated_at_once_has_no_controls(parallel_panel):
    """Все юниты получили воздействие в один месяц: колонка меняется только во времени, как «после».
    Это дизайн без контрольной группы — DiD, SC и SDID невозможны."""
    df = parallel_panel.assign(launch=parallel_panel["post"]).drop(columns=["treated_group", "post"])
    card = profile(df)
    d = card.design_for("launch")
    assert d is not None and d.n_control_units == 0 and d.n_treated_units == 40
    vs, flags, _ = verdicts(df, CausalQuery("effect", treatment="launch", outcome="y"))
    assert "no_controls" in flags
    assert {vs[m].status for m in ("did", "sdid", "synthetic_control")} == {FORBIDDEN}


def test_discovery_question(discovery):
    vs, flags, _ = verdicts(discovery, CausalQuery("discovery"))
    assert usable(vs) == {"pc", "ges", "lingam", "notears", "fci"}
    assert "non_gaussian" in flags  # равномерный шум
    assert vs["lingam"].pros


def test_not_causal_question_forbids_everything(rct):
    vs, _, _ = verdicts(rct, CausalQuery("not_causal"))
    assert usable(vs) == set()


def test_heterogeneity_question(rct):
    vs, _, _ = verdicts(rct, CausalQuery("heterogeneity", treatment="got_coupon", outcome="spend", randomized=True))
    assert {"meta_learners", "causal_forest", "dml", "doubly_robust"} <= usable(vs)
    assert vs["diff_in_means"].status == FORBIDDEN


def test_data_card_text_is_compact(smoking):
    text = profile(smoking).to_text()
    assert "structure=panel" in text and "design group_post(california,after_treatment)" in text
    assert len(text.splitlines()) == 1 + 1 + 9 + 1
