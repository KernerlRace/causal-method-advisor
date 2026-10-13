"""Этап 3: вопросы, целевые ответы и обучающий набор."""
import re

import numpy as np
import pytest

from datagen.build import build_one
from datagen.build_sft import examples_for, split_of
from datagen.explanations import CARDS, diag_text, explain, parse_fields
from datagen.questions import TEMPLATE_SETS, is_heldout, make_question


@pytest.fixture(scope="module")
def records():
    recs = [build_one((i, 99, None)) for i in range(60)]
    assert not [r for r in recs if "error" in r], [r["error"] for r in recs if "error" in r][:1]
    return recs


def _mentioned(rec, col, text):
    t = text.lower()
    if col.lower() in t:
        return True
    words = [w for w in re.findall(r"\w+", rec["concepts"].get(col, "").lower()) if len(w) > 3]
    return bool(words) and sum(w in t for w in words) >= max(1, len(words) // 2)


def test_questions_mention_treatment_and_outcome(records):
    total = ok = 0
    for rec in records:
        q = rec["query"]
        if q["question_type"] not in ("effect", "heterogeneity"):
            continue
        for k in range(3):
            text = make_question(rec, np.random.default_rng([rec["seed"], k]))["text"]
            total += 1
            ok += _mentioned(rec, q["treatment"], text) and _mentioned(rec, q["outcome"], text)
    assert ok / total >= 0.95, (ok, total)


def test_context_is_verbalized(records):
    """Сведения, которых нет в данных (инструмент, медиатор, скрытый фактор), попадают в вопрос."""
    for rec in records:
        q = rec["query"]
        for k in range(2):
            text = make_question(rec, np.random.default_rng([rec["seed"], k]))["text"]
            for role in ("instrument", "mediator"):
                if q.get(role):
                    assert _mentioned(rec, q[role], text), (role, text)


def test_parse_fields_match_query(records):
    for rec in records:
        f, q = parse_fields(rec), rec["query"]
        assert f["type"] == q["question_type"]
        for k, short in (("treatment", "treatment"), ("outcome", "outcome"), ("instrument", "instrument"),
                         ("mediator", "mediator"), ("running_variable", "running")):
            assert f.get(short) == q.get(k)


def test_explanation_names_the_right_method(records):
    for rec in records:
        for lang in ("en", "ru"):
            text = explain(rec, lang, np.random.default_rng(rec["seed"]))
            name = (lambda m: CARDS[m].name if lang == "en" else CARDS[m].name_ru)
            if rec["query"]["question_type"] == "not_causal":
                assert ("prediction question" in text) if lang == "en" else ("прогнозе" in text)
            elif rec["answer_mode"] == "refuse":
                assert ("cannot be identified" in text) if lang == "en" else ("нельзя идентифицировать" in text)
            else:
                best = max(rec["gold"], key=rec["gold"].get)
                assert name(best) in text, (rec["id"], lang, text)
                if rec["answer_mode"] == "options":
                    for m, w in rec["gold"].items():
                        if w >= 0.9:
                            assert name(m) in text


NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w%])")


def _card_numbers():
    nums = set()
    for c in CARDS.values():
        for v in list(c.requires.values()) + list(c.recommended.values()):
            if isinstance(v, (int, float)):
                nums.add(str(v))
        for txt in (c.name, c.name_ru, *c.assumptions, *c.assumptions_ru):
            nums |= set(NUM.findall(txt))
    return nums


def test_numbers_in_explanations_come_from_inputs(records):
    """Модель не должна учиться выдумывать цифры: всё, что есть в объяснении, есть во входе."""
    allowed_extra = _card_numbers()
    for rec in records:
        for lang in ("en", "ru"):
            rng = np.random.default_rng(rec["seed"])
            text = explain(rec, lang, rng)
            q = make_question(rec, np.random.default_rng(rec["seed"]), lang=lang)["text"]
            source = diag_text(rec) + "\n" + rec["card_text"] + "\n" + q
            src_nums = set(NUM.findall(source)) | allowed_extra
            for n in NUM.findall(text):
                assert n in src_nums, (rec["id"], n, text)


def test_heldout_templates_only_in_test(records):
    for rec in records:
        for ex in examples_for(rec, 3):
            if ex["heldout_template"]:
                assert ex["split"] == "test"
    heldout = [f"{l}:{k}:{i}" for (l, k), pool in TEMPLATE_SETS.items() for i in range(len(pool))
               if is_heldout(f"{l}:{k}:{i}")]
    assert 0.05 <= len(heldout) / sum(len(p) for p in TEMPLATE_SETS.values()) <= 0.2


def test_split_is_by_scenario():
    ids = [f"rct-{i}" for i in range(4000)]
    splits = [split_of(i) for i in ids]
    share = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert 0.86 < share["train"] < 0.94 and 0.03 < share["test"] < 0.07
    assert split_of("rct-1") == split_of("rct-1")


def test_gold_agrees_with_diagnostics(records):
    """После согласования эталон не противоречит своему же входу."""
    for rec in records:
        flags = {f for r in rec["diagnostics"].values() for f in r["flags"]}
        if not rec["gold"]:
            continue
        best = max(rec["gold"], key=rec["gold"].get)
        if "pretrend_violated" in flags:
            assert best != "did"
        if "gaussian_data" in flags and rec["query"]["question_type"] == "discovery":
            assert best != "lingam"
        for m, w in rec["gold"].items():
            if w >= 0.5:
                assert rec["verdicts"][m]["status"] != "forbidden", (rec["id"], m)


def test_examples_fit_in_context(records):
    lengths = [sum(len(ex[k]) for k in ("card", "question", "parse", "diag", "explanation"))
               for rec in records for ex in examples_for(rec, 2)]
    assert max(lengths) < 4000
