"""Обучение BPE на train-части SFT, карточках методов и открытом корпусе + отчёт по длинам.

  .venv/Scripts/python.exe -m model.train_tokenizer --vocab-size 16384 --corpus

Валидационные части в обучение не попадают: на них меряем символы на токен (EN и RU) — на примерах
задачи и на статьях корпуса — и длину целого обучающего примера: она должна влезать в контекст модели.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from advisor.knowledge import load_cards
from model.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = ROOT / "data" / "sft"
OUT = ROOT / "models" / "tokenizer.json"
FIELDS = ("card", "question", "parse", "diag", "explanation")


def read_jsonl(path: Path, limit: int | None = None) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)


def methods_text(rec: dict) -> str:
    probs = " ".join(f"<M:{m}> {p:.2f}" for m, p in sorted(rec["methods"].items(), key=lambda kv: -kv[1]))
    return f"<MODE:{rec['mode']}> {probs}"


def sequence(rec: dict) -> str:
    """Черновой формат целого примера — для оценки длины; окончательный формат задаёт этап 8."""
    return (f"<CARD>{rec['card']}<Q>{rec['question']}<PARSE>{rec['parse']}<DIAG>{rec['diag']}"
            f"<ANS>{methods_text(rec)}<EXPL>{rec['explanation']}<EOS>")


CORPUS_FOR_TOKENIZER = ("wiki_en.jsonl", "wiki_ru.jsonl", "facure.jsonl")  # own.jsonl повторяет SFT


def training_texts(max_examples: int, ru_weight: int, corpus: bool = False) -> Iterator[tuple[str, int]]:
    for rec in read_jsonl(SFT_DIR / "train.jsonl", max_examples):
        weight = ru_weight if rec["lang"] == "ru" else 1
        for field in FIELDS:
            yield rec[field], weight
        yield methods_text(rec), weight
    for card in load_cards().values():
        for text in (card.name, card.name_ru, card.summary, card.summary_ru,
                     *card.assumptions, *card.assumptions_ru):
            if text:
                yield text, 1
    if corpus:
        from datagen.corpus import corpus_docs
        for doc in corpus_docs("train", CORPUS_FOR_TOKENIZER):
            yield doc["text"], ru_weight if doc["lang"] == "ru" else 1


def report(tok: BPETokenizer, n_val: int) -> dict:
    chars = {"en": 0, "ru": 0}
    tokens = {"en": 0, "ru": 0}
    lengths: list[int] = []
    for rec in read_jsonl(SFT_DIR / "val.jsonl", n_val):
        text = rec["question"] + "\n" + rec["explanation"]
        ids = tok.encode(text, allow_special=False)
        assert tok.decode(ids) == text, "decode(encode(x)) != x"
        chars[rec["lang"]] += len(text)
        tokens[rec["lang"]] += len(ids)
        lengths.append(len(tok.encode(sequence(rec))))
    arr = np.array(lengths)
    from datagen.corpus import corpus_docs
    c_chars = {"en": 0, "ru": 0}
    c_tokens = {"en": 0, "ru": 0}
    for doc in corpus_docs("val", CORPUS_FOR_TOKENIZER):
        c_chars[doc["lang"]] += len(doc["text"])
        c_tokens[doc["lang"]] += len(tok.encode(doc["text"], allow_special=False))
    return {
        "chars_per_token": {lang: round(chars[lang] / max(tokens[lang], 1), 2) for lang in chars},
        "chars_per_token_corpus_val": {lang: round(c_chars[lang] / max(c_tokens[lang], 1), 2) for lang in c_chars},
        "sequence_tokens": {"p50": int(np.percentile(arr, 50)), "p95": int(np.percentile(arr, 95)),
                            "max": int(arr.max()), "share_over_1024": round(float((arr > 1024).mean()), 4)},
        "examples": len(lengths),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--max-examples", type=int, default=30000, help="сколько train-примеров брать для слияний")
    ap.add_argument("--ru-weight", type=int, default=2, help="вес русских примеров: их в данных ~16%")
    ap.add_argument("--val-examples", type=int, default=2000)
    ap.add_argument("--corpus", action="store_true", help="добавить статьи открытого корпуса (data/corpus)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    tok = BPETokenizer.train(training_texts(args.max_examples, args.ru_weight, args.corpus), args.vocab_size)
    print(f"trained: vocab={tok.vocab_size} merges={len(tok.merges)} in {time.time() - t0:.0f}s")
    tok.save(args.out)
    print(f"saved: {args.out}")
    print(json.dumps(report(tok, args.val_examples), indent=2))


if __name__ == "__main__":
    main()
