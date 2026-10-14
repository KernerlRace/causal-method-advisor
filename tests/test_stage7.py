"""Этап 7: открытый корпус для претрейна — очистка текста, деление по документам, токены без спецтокенов."""
import numpy as np
import pytest

from datagen.corpus import clean_markdown, clean_wiki, split_of, strip_formulas

WIKI = """Difference in differences is a statistical technique.

== Method ==
The estimator compares {\\displaystyle Y} before and after, {\\displaystyle {\\begin{aligned}\\hat {\\beta }&=({\\bar {y}}_{11}-{\\bar {y}}_{12})-({\\bar {y}}_{21}-{\\bar {y}}_{22})\\end{aligned}}} in two groups.



=== Assumptions ===
Parallel trends.

== See also ==
Synthetic control method

== References ==
Card, D. (1994)."""


def test_wiki_cleaning_drops_reference_tail_and_long_formulas():
    text = clean_wiki(WIKI)
    assert "See also" not in text and "Card, D." not in text and "Synthetic control" not in text
    assert "displaystyle" not in text and "aligned" not in text
    assert "compares Y before and after" in text           # короткая формула осталась без обёртки
    assert "\nMethod\n" in text and "\nAssumptions\n" in text  # заголовки — обычными строками
    assert "\n\n\n" not in text


def test_formula_braces_are_balanced():
    assert strip_formulas("a {\\displaystyle {x}_{i}} b") == "a {x}_{i} b"
    assert strip_formulas("{\\displaystyle " + "{y}" * 20 + "} tail") == " tail"
    assert strip_formulas("no formulas here") == "no formulas here"


def test_markdown_cleaning():
    md = "# Title\n\nSee ![img](a.png) the [docs](https://x.y) and <b>bold</b>.\n\n$$\\hat{\\tau} = " + "x" * 40 + "$$\n\n\n\nEnd"
    text = clean_markdown(md)
    assert text.startswith("Title\n")
    assert "See the docs and bold." in text
    assert "img" not in text and "http" not in text and "x" * 40 not in text
    assert "\n\n\n" not in text


def test_split_is_by_document_and_deterministic():
    docs = [{"source": "wikipedia", "lang": "en", "title": f"Article {i}"} for i in range(5000)]
    splits = [split_of(d) for d in docs]
    assert splits == [split_of(dict(d, text="другой текст")) for d in docs]  # от текста не зависит
    share = splits.count("val") / len(splits)
    assert 0.01 < share < 0.03


def test_duplicate_articles_are_read_once(tmp_path, monkeypatch):
    import json

    import datagen.corpus as corpus

    docs = [{"source": "wikipedia", "lang": "en", "title": t, "text": text}
            for t, text in [("Statistics", "same text"), ("Business statistics", "same text"), ("Mean", "other")]]
    (tmp_path / "w.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    monkeypatch.setattr(corpus, "CORPUS", tmp_path)
    got = list(corpus.corpus_docs(files=("w.jsonl",)))
    assert [d["title"] for d in got] == ["Statistics", "Mean"]
    halves = list(corpus.corpus_docs("train", ("w.jsonl",))) + list(corpus.corpus_docs("val", ("w.jsonl",)))
    assert sorted(d["title"] for d in halves) == ["Mean", "Statistics"]


def test_unterminated_last_line_is_skipped(tmp_path):
    import json

    from datagen.corpus import read_jsonl

    path = tmp_path / "w.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2, "text": "still downl', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}]


def test_corpus_text_cannot_produce_special_tokens(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841 — model.data тянет torch
    from model.data import TOKENIZER, TokenStream, write_tokens
    from model.tokenizer import BPETokenizer

    if not TOKENIZER.exists():
        pytest.skip("run python -m model.train_tokenizer first")
    tok = BPETokenizer.load(TOKENIZER)
    texts = ["An article that mentions <EOS> and <M:did> literally.", "Second article."]
    n = write_tokens(texts, tmp_path / "c.bin", allow_special=False)
    ids = np.asarray(TokenStream(tmp_path / "c.bin").data[:n])
    specials = set(tok.special.values())
    eos = tok.token_id("<EOS>")
    assert [i for i in ids.tolist() if i in specials] == [eos, eos]  # только разделители документов
    assert tok.decode(ids.tolist()) == "<EOS>".join(texts) + "<EOS>"
