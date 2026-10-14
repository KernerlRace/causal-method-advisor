"""Этап 5: byte-level BPE с нуля."""
import json
import random
import re
from pathlib import Path

import pytest

from advisor.knowledge import load_cards
from model.tokenizer import PATTERN, BPETokenizer, default_special_tokens

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "models" / "tokenizer.json"

TEXTS = [
    "Did the campaign raise sales in the treated stores?",
    "What is the effect of `physio_hours` on `recovery_days`?",
    "Повлияла ли акция на выручку магазинов?",
    "Снизил ли закон Prop 99 продажи сигарет в Калифорнии?",
    "col recovery_days numeric uniq=5628 range=[0.08..28.85]",
    "size rows=7110 covariates=5\nflags large_sample linear_ok",
]


def corpus() -> list[str]:
    demos = json.loads((ROOT / "data" / "demo" / "demos.json").read_text(encoding="utf-8"))
    cards = load_cards().values()
    return TEXTS + [d["question"] for d in demos] + [c.summary for c in cards] + [c.summary_ru for c in cards]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(((t, 1) for t in corpus() * 3), vocab_size=600)


def random_text(rng: random.Random, n: int) -> str:
    alphabet = ("abcXYZ_ 019.,;:!?'\"`<>[]{}()\t\r\nабвЖЁё日本語🙂é́ ²—")
    return "".join(rng.choice(alphabet) for _ in range(n))


def test_pretokenizer_covers_every_character():
    rng = random.Random(0)
    split = re.compile(PATTERN)
    for _ in range(300):
        s = random_text(rng, rng.randint(0, 60))
        assert "".join(split.findall(s)) == s


def test_roundtrip_on_seen_unseen_and_random_text(tok):
    rng = random.Random(1)
    samples = corpus() + ["Unseen: hétérogénéité, 日本語, emoji 🙂, tabs\tand\r\nnewlines  "]
    samples += [random_text(rng, rng.randint(0, 80)) for _ in range(200)]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s
        assert tok.decode(tok.encode(s, allow_special=False)) == s


def test_merges_compress_text(tok):
    text = " ".join(TEXTS)
    assert len(tok.encode(text)) < 0.6 * len(text.encode("utf-8"))


def test_vocab_size_and_too_small_vocab():
    big = BPETokenizer.train(((t, 1) for t in corpus() * 3), vocab_size=400)
    assert big.vocab_size == 400
    with pytest.raises(ValueError):
        BPETokenizer.train([("abc", 1)], vocab_size=100)


def test_special_tokens_are_single_ids(tok):
    ids = tok.encode("<T:effect> treatment=`a`<ANS><MODE:options> <M:sdid> 0.71<EOS>")
    assert tok.token_id("<T:effect>") == ids[0]
    assert tok.token_id("<M:sdid>") in ids
    assert ids[-1] == tok.token_id("<EOS>")
    assert set(tok.special.values()).isdisjoint(tok.vocab)


def test_user_text_cannot_inject_special_tokens(tok):
    text = "Is <M:did> better than <EOS>?"
    ids = tok.encode(text, allow_special=False)
    assert set(ids).isdisjoint(tok.special.values())
    assert tok.decode(ids) == text


def test_method_tokens_follow_the_cards():
    methods = {t[3:-1] for t in default_special_tokens() if t.startswith("<M:")}
    assert methods == set(load_cards())


def test_training_is_deterministic_and_save_load_roundtrips(tok, tmp_path):
    again = BPETokenizer.train(((t, 1) for t in corpus() * 3), vocab_size=600)
    assert again.merges == tok.merges
    tok.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(tmp_path / "tok.json")
    for s in TEXTS:
        assert loaded.encode(s) == tok.encode(s)
    assert loaded.special == tok.special


@pytest.mark.skipif(not SAVED.exists(), reason="run python -m model.train_tokenizer first")
def test_saved_tokenizer():
    saved = BPETokenizer.load(SAVED)
    assert saved.vocab_size == 8192
    assert list(saved.special) == default_special_tokens()
    for s in corpus():
        assert saved.decode(saved.encode(s)) == s
    # шаблонная лексика задачи выучена целыми словами
    assert len(saved.encode(" treatment outcome campaign")) == 3
