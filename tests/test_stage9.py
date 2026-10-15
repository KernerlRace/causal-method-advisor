"""Этап 9: калиброванная уверенность, режим ответа, отказы.
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
