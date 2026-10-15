"""Этап 9: калиброванная уверенность, режим ответа, отказы, ловушки.
Математика калибровки и решение о режиме — без PyTorch; нейробэкенд — на крошечной модели (CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advisor.confidence import (Calibration, best_set_target, decide, file_sha, fit_temperature, reliability,
                                softmax)
from advisor.knowledge import load_cards
from advisor.pipeline import advise

ROOT = Path(__file__).resolve().parent.parent
METHODS = list(load_cards())
MODES = {"recommend": 0.0, "options": 0.0, "refuse": 0.0}


def logits(**high) -> dict[str, float]:
    z = {m: -5.0 for m in METHODS}
    z.update(high)
    return z


def token(mode: str) -> dict[str, float]:
    return {**MODES, mode: 5.0}


# ---------- калибровка ----------

def test_temperature_fit_recovers_the_true_temperature():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, (4000, 6))
    p = softmax(z, 2.0)
    y = np.array([rng.choice(6, p=row) for row in p])
    assert fit_temperature(z, np.eye(6)[y]) == pytest.approx(2.0, rel=0.1)
    assert softmax(z, 1.0).sum(1) == pytest.approx(np.ones(len(z)))


def test_reliability_and_ece():
    ece, pts = reliability(np.full(10, 0.9), np.array([1] * 9 + [0]))
    assert ece == pytest.approx(0.0) and pts[0]["n"] == 10 and pts[0]["acc"] == pytest.approx(0.9)
    ece, _ = reliability(np.full(10, 0.9), np.zeros(10))
    assert ece == pytest.approx(0.9)
    ece, pts = reliability(np.array([0.2, 0.2, 0.8, 0.8]), np.array([0, 1, 1, 1]))
    assert len(pts) == 2 and ece == pytest.approx(0.5 * 0.3 + 0.5 * 0.2)


def test_target_splits_ties_between_best_methods():
    q = best_set_target({"dml": 1.0, "doubly_robust": 1.0, "causal_forest": 0.6}, METHODS)
    assert q[METHODS.index("dml")] == q[METHODS.index("doubly_robust")] == 0.5 and q.sum() == pytest.approx(1)


# ---------- режим ответа ----------

CAL = Calibration(method_T=1.0, mode_T=1.0, min_top=0.5, min_gap=0.3, rule="confidence")


def test_one_clear_method_is_recommended():
    d = decide(logits(did=5.0, sdid=2.0), token("options"), set(), CAL)
    assert d.mode == "recommend" and d.token_mode == "options"
    assert next(iter(d.probs)) == "did" and d.confidence > 0.9 and sum(d.probs.values()) == pytest.approx(1)


def test_two_equal_methods_give_options():
    d = decide(logits(dml=4.0, doubly_robust=4.0), token("recommend"), set(), CAL)
    assert d.mode == "options" and d.gap == pytest.approx(0.0, abs=1e-9)


def test_sensitivity_analysis_on_top_is_a_refusal():
    d = decide(logits(sensitivity_analysis=6.0), token("recommend"), set(), CAL)
    assert d.mode == "refuse"


def test_mass_on_forbidden_methods_lowers_confidence():
    """Модель ставила на DiD, правила его запретили: оставшийся ITS не выглядит уверенным выбором."""
    d = decide(logits(did=6.0, its=3.0), token("recommend"), {"did"}, CAL)
    assert next(iter(d.probs)) == "its" and d.probs["its"] > 0.9  # для показа — перенормированы
    assert d.confidence < 0.1 and d.mode == "options"              # уверенность — нет


def test_everything_forbidden_is_a_refusal():
    d = decide(logits(did=6.0), token("recommend"), set(METHODS), CAL)
    assert d.mode == "refuse" and d.probs == {} and d.confidence is None


def test_rules_token_and_either():
    z = logits(dml=4.0, doubly_robust=4.0)
    as_token = Calibration(rule="token")
    assert decide(z, token("recommend"), set(), as_token).mode == "recommend"
    either = Calibration(min_top=0.5, min_gap=0.3, rule="either")
    assert decide(z, token("recommend"), set(), either).mode == "options"
    assert decide(logits(did=6.0), token("refuse"), set(), either).mode == "refuse"


def test_temperature_changes_confidence_not_the_shown_scores():
    z = logits(did=2.0, sdid=1.0)
    sharp = Calibration(method_T=0.3, rule="token")
    a, b = decide(z, token("recommend"), set(), Calibration()), decide(z, token("recommend"), set(), sharp)
    assert a.probs == b.probs and b.confidence > a.confidence


def test_calibration_file_belongs_to_one_model(tmp_path):
    model, other = tmp_path / "model.pt", tmp_path / "other.pt"
    model.write_bytes(b"weights")
    other.write_bytes(b"other weights")
    cal = Calibration(0.4, 0.9, 0.55, 0.2, "either", model_sha=file_sha(model))
    path = tmp_path / "calibration.json"
    cal.save(path)
    assert Calibration.load(path, model=model) == cal
    assert Calibration.load(path, model=other) == Calibration()  # чужая модель — без калибровки
    assert Calibration.load(tmp_path / "missing.json") == Calibration()


def test_app_side_modules_do_not_import_torch():
    """Демо в облаке работает без PyTorch: пайплайн и калибровка не должны тянуть его при импорте."""
    code = "import sys, advisor.confidence, advisor.pipeline; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# ---------- отказ без скрытого конфаундера ----------

class _NoModelExplanation:
    name = "stub"

    def parse(self, question, card_text):
        return {"type": "effect", "treatment": "new_design", "outcome": "open_rate"}

    def rank(self, structure, diagnostics, verdicts, fields, question="", card_text=""):
        allowed = [m for m, v in verdicts.items() if v["status"] != "forbidden"]
        return {m: 1 / len(allowed) for m in allowed}, "recommend"


def test_refusal_without_hidden_confounder_explains_what_is_missing():
    """Один ряд, 5 недель до и 5 после: ITS нужно не меньше 8 периодов до, других методов для ряда нет."""
    weeks = pd.date_range("2025-03-03", periods=10, freq="W-MON").strftime("%Y-%m-%d")
    df = pd.DataFrame({"week": weeks, "new_design": [0] * 5 + [1] * 5,
                       "open_rate": [0.21, 0.212, 0.209, 0.215, 0.214, 0.23, 0.228, 0.233, 0.231, 0.235]})
    a = advise(df, "Did the new design raise the open rate?", _NoModelExplanation())
    assert a.mode == "refuse" and not a.methods
    assert "needs at least 8 pre-treatment periods" in a.explanation
    assert "confounder" not in a.explanation


# ---------- нейробэкенд на крошечной модели ----------

def _tiny_model(tmp_path) -> Path:
    torch = pytest.importorskip("torch")
    from model.data import TOKENIZER
    from model.tokenizer import BPETokenizer
    from model.transformer import GPT, GPTConfig

    torch.manual_seed(0)
    vocab = BPETokenizer.load(TOKENIZER).vocab_size
    path = tmp_path / "tiny.pt"
    GPT(GPTConfig(vocab_size=vocab, context=512, n_layer=1, n_head=2, d_model=32)).save(path)
    return path


def test_advisor_scores_feed_the_decision(tmp_path):
    from model.decode import Advisor

    adv = Advisor(_tiny_model(tmp_path), "cpu")
    card = "rows=500 cols=2 structure=cross_section\ncol y numeric uniq=480 range=[1..90]\ncol t binary uniq=2 values=0|1"
    fields = {"type": "effect", "treatment": "t", "outcome": "y"}
    diag = "size rows=500 covariates=0\nflags -\nallowed regression\nwarning -"
    modes, methods = adv.scores("Did t raise y?", card, fields, diag)
    assert set(modes) == {"recommend", "options", "refuse"} and set(methods) == set(METHODS)
    probs, mode, mode_p = adv.rank("Did t raise y?", card, fields, diag)
    assert mode == max(modes, key=modes.get)  # без калибровки — токен модели, как на этапе 8
    d = decide(methods, modes, set(), Calibration())
    assert probs == d.probs and mode_p == pytest.approx(d.mode_p)


def test_neural_backend_reports_calibrated_confidence_only_for_its_own_model(tmp_path, rct):
    from advisor.pipeline import NeuralBackend

    model = _tiny_model(tmp_path)
    path = tmp_path / "calibration.json"
    Calibration(0.5, 1.0, 0.3, 0.1, "either", model_sha=file_sha(model)).save(path)
    q = "Did the coupon raise spend?"
    fields = {"type": "effect", "treatment": "got_coupon", "outcome": "spend", "randomized": "yes"}
    nb = NeuralBackend(model, "cpu", calibration=path)
    assert nb.calibrated
    a = advise(rct, q, nb, fields=fields)
    assert a.mode in ("recommend", "options", "refuse")
    assert (a.confidence is not None) == bool(a.methods) and nb.last.token_mode in ("recommend", "options", "refuse")
    Calibration(model_sha="something else").save(path)
    plain = NeuralBackend(model, "cpu", calibration=path)
    assert not plain.calibrated and advise(rct, q, plain, fields=fields).confidence is None


# ---------- выбор порогов на val (eval/calibrate.py) ----------

def test_threshold_search_finds_a_separating_rule(tmp_path):
    from eval.calibrate import Scores, allowed_methods, search_thresholds

    assert allowed_methods("flags x\nallowed did sdid\nwarning -") == {"did", "sdid"}
    rng = np.random.default_rng(1)
    names = ["did", "sdid", "sensitivity_analysis"]
    n = 300
    kind = rng.choice(["recommend", "options", "refuse"], n)
    z = np.full((n, 3), -4.0)
    z[kind == "recommend", 0] = 4.0
    z[kind == "options", 0] = z[kind == "options", 1] = 3.0
    z[kind == "refuse", 2] = 4.0
    gold = np.where(z > 0, 1.0, 0.0)
    mode_logits = np.zeros((n, 3))
    np.savez(tmp_path / "s.npz", mode_logits=mode_logits, method_logits=z, allowed=np.ones((n, 3), bool),
             gold=gold, names=np.array(names), gold_mode=kind, qtype=np.array(["effect"] * n),
             scenario=kind, ckpt="x", model_sha="0")
    bal, a, g = search_thresholds(Scores(tmp_path / "s.npz"), 1.0, "confidence")
    assert bal == pytest.approx(1.0) and (a > 0.5 or g > 0)


# ---------- ловушки ----------

def test_trap_questions_match_their_data():
    from advisor.profiler import load_csv

    qs = [json.loads(line) for line in (ROOT / "eval" / "trap_questions.jsonl").open(encoding="utf-8")]
    assert len({q["id"] for q in qs}) == len(qs) == 48
    kinds = {q["trap"] for q in qs}
    assert kinds == {"unidentifiable", "no_controls", "prediction", "poor_overlap"}
    assert all(sum(q["trap"] == k for q in qs) >= 10 for k in kinds)
    cols = {}
    for q in qs:
        if q["file"] not in cols:
            cols[q["file"]] = set(load_csv(ROOT / "data" / q["file"]).columns)
        for k in ("target", "treatment", "outcome"):
            if k in q["fields"]:
                assert q["fields"][k] in cols[q["file"]], (q["id"], k)
        assert set(q["expect"]) <= {"refuse", "options"}


def test_trap_datasets_are_what_they_claim():
    from advisor.diagnostics import run_diagnostics
    from advisor.pipeline import query_from_fields
    from advisor.profiler import load_csv, profile

    flags = {}
    for name, t, y in (("card_offer_poor_overlap", "premium_card_offer", "card_spend"),
                       ("icu_poor_overlap", "icu_admission", "length_of_stay"),
                       ("stores_all_treated", "self_checkout", "revenue"),
                       ("newsletter_short_series", "new_design", "open_rate")):
        df = load_csv(ROOT / "data" / "traps" / f"{name}.csv")
        card = profile(df)
        q = query_from_fields({"type": "effect", "treatment": t, "outcome": y}, list(df.columns))
        flags[name] = {f for r in run_diagnostics(df, card, q).values() for f in r.flags}
    assert "poor_overlap" in flags["card_offer_poor_overlap"] and "poor_overlap" in flags["icu_poor_overlap"]
    assert "running_variable_found" not in flags["card_offer_poor_overlap"] | flags["icu_poor_overlap"]
    assert "no_controls" in flags["stores_all_treated"] and "no_controls" in flags["newsletter_short_series"]


# ---------- уверенность в разборе (проход 1) ----------

PARSE_CARD = ("rows=500 cols=4 structure=cross_section\n"
              "col sales numeric uniq=480 range=[1..90]\n"
              "col promo binary uniq=2 values=yes|no\n"
              "col region categorical uniq=3 values=n|s|e\n"
              "col sales_lag numeric uniq=470 range=[1..88]")


def _scores(type_lp=-1e-6, treat=-1e-6, out=-1e-6, off=0.0, qtype="effect"):
    from advisor.confidence import FieldScore, ParseScores

    fields = {k: FieldScore(0.0, -1e-3, v, v + off, 4) for k, v in (("treatment", treat), ("outcome", out))}
    return ParseScores(qtype, {"effect": 5.0, "heterogeneity": 1.0, "discovery": 0.0, "not_causal": 0.0},
                       type_lp, type_lp, fields if qtype != "discovery" else {})


def test_parse_scores_round_trip_and_weakest_part():
    from advisor.confidence import ParseScores, parse_features

    s = _scores(type_lp=-0.01, treat=-2.0, out=-0.1)
    assert ParseScores.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert s.parts() == {"type": -0.01, "treatment": -2.0, "outcome": -0.1} and s.weakest() == "treatment"
    assert _scores(type_lp=-3.0).weakest() == "type"
    assert _scores(qtype="discovery").parts() == {"type": -1e-6}
    # больше неуверенности — больше признак log(−log p); масса вне колонок CSV — отдельный признак
    lo, hi = parse_features(_scores(treat=-1e-6), "parts_loglog"), parse_features(_scores(treat=-0.5), "parts_loglog")
    assert hi[1] > lo[1] and hi[0] == lo[0]
    assert parse_features(_scores(off=-3.0), "parts_loglog")[2] > parse_features(_scores(), "parts_loglog")[2]
    assert parse_features(s, "joint")[0] == pytest.approx(-2.11)


def test_logistic_fit_auroc_and_log_loss():
    from advisor.confidence import auroc, fit_logistic, log_loss, sigmoid

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 2))
    y = rng.random(20000) < sigmoid(X @ np.array([2.0, -1.0]) + 0.5)
    w, b = fit_logistic(X, y, l2=1e-3)
    assert w == pytest.approx([2.0, -1.0], abs=0.1) and b == pytest.approx(0.5, abs=0.1)
    assert auroc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert auroc(np.array([0.1, 0.9]), np.array([1, 0])) == 0.0
    assert auroc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5 and auroc(np.ones(3), np.ones(3)) is None
    assert log_loss(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(np.log(2))


def test_parse_confidence_needs_a_parse_calibration():
    from advisor.confidence import parse_confidence

    s = _scores(treat=-0.7)
    assert parse_confidence(s, Calibration()) is None and parse_confidence(None, Calibration()) is None
    cal = Calibration(parse_kind="joint", parse_w=[2.0], parse_b=3.0)
    assert parse_confidence(s, cal) == pytest.approx(1 / (1 + np.exp(-(3.0 + 2.0 * (-0.7 - 2e-6)))))
    assert parse_confidence(_scores(treat=-5.0), cal) < parse_confidence(s, cal)


def _value_tree(adv, cache, logits, seqs, i=0) -> dict:
    """Эталон для _choose: вероятность каждого варианта при пошаговом выборе среди совместимых токенов."""
    import torch

    if len(seqs) == 1:
        return {next(iter(seqs)): 1.0}
    allowed = sorted({s[i] for s in seqs.values() if len(s) > i})
    p = torch.softmax(logits.double()[allowed], -1)
    out = {}
    for t, pt in zip(allowed, p.tolist()):
        sub = {v: s for v, s in seqs.items() if len(s) > i and s[i] == t}
        branch = cache.fork()
        nxt = adv._feed(branch, [t]) if len(sub) > 1 else None
        out.update({v: pt * q for v, q in _value_tree(adv, branch, nxt, sub, i + 1).items()})
    return out


def test_parse_records_the_probability_of_every_choice(tmp_path):
    import torch
    from model.decode import Advisor, card_columns

    adv = Advisor(_tiny_model(tmp_path), "cpu")  # случайные веса: вероятности не крайние
    for q in ("Did the promo raise sales?", "Forecast sales", "What causes what?"):
        fields = adv.parse(q, PARSE_CARD)
        s = adv.last_parse
        assert s.type == fields["type"] and set(s.fields) == set(fields) - {"type"}
        p_type = torch.softmax(torch.tensor(list(s.type_logits.values()), dtype=torch.float64), -1).max()
        assert np.exp(s.type_lp) == pytest.approx(float(p_type), rel=1e-4) and s.type_lp_full <= s.type_lp <= 0
        for f in s.fields.values():
            assert f.key_lp_full <= f.key_lp <= 0 and f.value_lp_full <= f.value_lp + 1e-9 <= 1e-9 and f.n >= 1
        assert s.end_lp <= 0
    # жадная вероятность значения — распределение по колонкам (sales и sales_lag делят первые токены)
    cache, logits = adv._start(adv.pr.pass1(PARSE_CARD, "Did the promo raise sales?"))
    n = len(adv.pr.text(" treatment"))
    seqs = {c: adv.pr.text(f" treatment=`{c}`")[n:] for c in card_columns(PARSE_CARD)}
    value, lp, _ = adv._choose(cache, logits, seqs)
    tree = _value_tree(adv, cache, logits, seqs)
    assert sum(tree.values()) == pytest.approx(1.0) and np.exp(lp) == pytest.approx(tree[value], rel=1e-4)


def test_advice_asks_to_confirm_an_uncertain_parse(tmp_path, rct):
    from advisor.pipeline import NeuralBackend

    model = _tiny_model(tmp_path)
    path = tmp_path / "calibration.json"
    q = "Did the coupon raise spend?"
    # порог 1.0: попросить подтвердить любой разбор; 0.0 — никогда
    for min_parse, asks in ((1.0, True), (0.0, False)):
        Calibration(0.5, 1.0, 0.3, 0.1, "either", model_sha=file_sha(model), parse_kind="joint", parse_w=[1.0],
                    parse_b=2.0, min_parse=min_parse).save(path)
        nb = NeuralBackend(model, "cpu", calibration=path)
        a = advise(rct, q, nb)
        assert 0 <= a.parse_confidence <= 1
        assert (a.confirm is not None) == asks
        if asks:
            assert a.confirm == nb.last_parse.weakest() and a.confirm in ("type", "treatment", "outcome", "target")
    fields = {"type": "effect", "treatment": "got_coupon", "outcome": "spend"}
    confirmed = advise(rct, q, nb, fields=fields)  # роли задал или подтвердил пользователь
    assert confirmed.parse_confidence is None and confirmed.confirm is None
    Calibration(model_sha="another model").save(path)
    a = advise(rct, q, NeuralBackend(model, "cpu", calibration=path))
    assert a.parse_confidence is None and a.confirm is None


def _parse_runs(path: Path, wrong_share: dict[str, float], n: int = 600) -> None:
    rng = np.random.default_rng(2)
    for split, share in wrong_share.items():
        with (path / f"parse_{split}.jsonl").open("w", encoding="utf-8") as f:
            for i in range(n):
                right = rng.random() >= share
                treat = -rng.exponential(1e-4) if right else -rng.exponential(1.0)
                gold = {"type": "effect", "treatment": "t", "outcome": "y"}
                pred = gold if right else {**gold, "treatment": "x"}
                f.write(json.dumps({"id": f"s{i // 2}#q{i % 2}", "gold": gold, "pred": pred,
                                    "scores": _scores(treat=treat).to_dict()}) + "\n")
        (path / f"parse_{split}.meta.json").write_text(json.dumps({"split": split, "model_sha": "m"}))


def test_parse_calibration_is_fitted_on_val_only(tmp_path, monkeypatch):
    """fit_parse: признаки выбираются по CV на val, порог — на val; на test только отчёт."""
    import eval.calibrate as calib
    from advisor.confidence import PARSE_KINDS

    _parse_runs(tmp_path, {"val": 0.1, "test": 0.1})
    monkeypatch.setattr(calib, "RUNS", tmp_path)
    cal = Calibration(model_sha="m")
    report = calib.fit_parse(cal)
    assert cal.parse_kind in PARSE_KINDS and cal.parse_kind == report["kind"] and len(cal.parse_w) >= 1
    assert report["test"]["calibrated"]["auroc"] > 0.95 and 0 < cal.min_parse < 1
    assert report["test"]["flag"]["wrong_flagged"] > 0.8 and report["test"]["flag"]["right_flagged"] < 0.2
    with pytest.raises(SystemExit):
        calib.fit_parse(Calibration(model_sha="another model"))


def test_too_few_wrong_parses_fall_back_to_the_models_own_probability(tmp_path, monkeypatch):
    """На знакомых шаблонах модель почти не ошибается: подбирать не на чем, порог — 1% ложных тревог на val."""
    import eval.calibrate as calib
    from advisor.confidence import parse_confidence

    _parse_runs(tmp_path, {"val": 0.0, "test": 0.2})
    monkeypatch.setattr(calib, "RUNS", tmp_path)
    cal = Calibration(model_sha="m")
    report = calib.fit_parse(cal)
    assert report["val_wrong_parses"] == 0 and cal.parse_kind == "raw" and cal.parse_w == []
    assert report["val"]["flag"]["flagged"] == pytest.approx(calib.FALSE_ALARMS, abs=0.005)
    assert report["test"]["calibrated"]["auroc"] > 0.95
    s = _scores(treat=-0.3)
    assert parse_confidence(s, cal) == pytest.approx(np.exp(sum(s.parts().values())))
