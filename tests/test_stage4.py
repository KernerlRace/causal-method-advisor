"""Этап 4: бейзлайн, сквозной пайплайн и интерфейс."""
import pytest

from advisor.pipeline import (MODELS_DIR, AdviceError, advise, detect_lang, query_from_fields,
                              weights_for_explanation)
from advisor.profiler import load_csv

needs_model = pytest.mark.skipif(not (MODELS_DIR / "baseline.joblib").exists(),
                                 reason="run scripts/train_baseline.py first")


@pytest.fixture(scope="module")
def backend():
    from advisor.pipeline import BaselineBackend
    return BaselineBackend()


def test_detect_lang():
    assert detect_lang("Did `promo` increase `sales`?") == "en"
    assert detect_lang("Повлиял ли `promo` на `sales`?") == "ru"
    assert detect_lang("What is the effect of `акция` on `выручка`?") == "en"


def test_query_from_fields_validates_columns():
    cols = ["a", "b"]
    with pytest.raises(AdviceError):
        query_from_fields({"type": "effect", "treatment": "a"}, cols)
    with pytest.raises(AdviceError):
        query_from_fields({"type": "effect", "treatment": "a", "outcome": "zzz"}, cols)
    with pytest.raises(AdviceError):
        query_from_fields({"type": "effect", "treatment": "a", "outcome": "a"}, cols)
    q = query_from_fields({"type": "effect", "treatment": "a", "outcome": "b", "randomized": "no"}, cols)
    assert q.randomized is False


def test_weights_follow_mode():
    probs = {"sdid": 0.6, "synthetic_control": 0.3, "did": 0.1}
    assert sum(w >= 0.9 for w in weights_for_explanation(probs, "options").values()) == 2
    w = weights_for_explanation({"a": 0.5, "b": 0.48}, "recommend")
    assert w["a"] == 1.0 and w["b"] < 0.9


@needs_model
def test_smoking_end_to_end(backend):
    df = load_csv("data/demo/smoking.csv")
    a = advise(df, "Did Proposition 99 (the `california` law) reduce cigarette sales (`cigsale`)?", backend)
    assert a.fields["treatment"] == "california" and a.fields["outcome"] == "cigsale"
    assert set(list(a.methods)[:2]) <= {"sdid", "synthetic_control"}
    assert "did" not in list(a.methods)[:1]
    assert "Synthetic" in a.explanation and "-22.2" in a.explanation


@needs_model
def test_manual_roles_override(backend):
    """Пользователь поправил роли — второй проход пересчитывается по ним."""
    df = load_csv("data/demo/tax_instrument.csv")
    fields = {"type": "effect", "treatment": "tax_rate", "outcome": "cigsale", "instrument": "neighbor_policy",
              "hidden": "yes"}
    a = advise(df, "Does a higher tax rate reduce cigarette sales?", backend, fields=fields)
    assert next(iter(a.methods)) == "iv"
    assert "first-stage F" in a.explanation


@needs_model
def test_not_causal_question(backend):
    df = load_csv("data/demo/marketing_ab_test.csv")
    a = advise(df, "Which features are most predictive of `conversion_rate`?", backend)
    assert a.mode == "refuse" and "prediction question" in a.explanation


@needs_model
def test_all_demos_give_expected_family(backend):
    import json
    expected = {"marketing_ab_test.csv": {"diff_in_means", "regression"}, "training_threshold.csv": {"rdd"},
                "drug_rollout_panel.csv": {"sdid", "synthetic_control"}, "medicaid_timeseries.csv": {"its"},
                "push_unidentifiable.csv": {"sensitivity_analysis"}, "remote_work_frontdoor.csv": {"front_door"},
                "app_metrics_discovery.csv": {"lingam", "pc", "ges"}}
    for d in json.load(open("data/demo/demos.json", encoding="utf-8")):
        if d["file"] in expected:
            a = advise(load_csv("data/demo/" + d["file"]), d["question"], backend)
            assert next(iter(a.methods)) in expected[d["file"]], (d["file"], a.methods)


@needs_model
def test_streamlit_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Several" in m.value or "Recommend" in m.value for m in at.markdown)
