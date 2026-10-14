"""Токены на диске: тексты → один поток uint16 с <EOS> после каждого документа → случайные окна.

  .venv/Scripts/python.exe -m model.data sft --splits train val

Словарь 8192 < 65536, поэтому uint16: 36M токенов SFT занимают ~70 МБ. Поток читается через memmap,
так что корпус претрейна не обязан влезать в память.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from model.tokenizer import BPETokenizer
from model.train_tokenizer import SFT_DIR, read_jsonl, sequence

ROOT = Path(__file__).resolve().parent.parent
TOKENS_DIR = ROOT / "data" / "tokens"
TOKENIZER = ROOT / "models" / "tokenizer.json"


def write_tokens(texts: Iterable[str], out: Path, tokenizer: Path = TOKENIZER) -> int:
    """Кодирует тексты (спецтокены разметки распознаются) и дописывает <EOS> после каждого."""
    tok = BPETokenizer.load(tokenizer)
    if tok.vocab_size > 65536:
        raise ValueError("словарь не влезает в uint16")
    eos = tok.token_id("<EOS>")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("wb") as f:
        for batch in _batched(texts, 1024):
            ids: list[int] = []
            for text in batch:
                ids.extend(tok.encode(text))
                ids.append(eos)
            f.write(np.asarray(ids, dtype=np.uint16).tobytes())
            n += len(ids)
    return n


def _batched(texts: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class TokenStream:
    """Случайные окна из потока токенов: x = окно, y = то же окно со сдвигом на один токен."""

    def __init__(self, path: Path):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, context: int, rng: np.random.Generator,
              device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.data) <= context:
            raise ValueError(f"в {self.path.name} {len(self.data)} токенов — меньше окна {context}+1")
        starts = rng.integers(0, len(self.data) - context - 1, batch_size)
        chunk = np.stack([self.data[s:s + context + 1] for s in starts]).astype(np.int64)
        xy = torch.from_numpy(chunk)
        if torch.device(device).type == "cuda":
            xy = xy.pin_memory().to(device, non_blocking=True)
        return xy[:, :-1], xy[:, 1:]


def sft_texts(split: str, limit: int | None = None) -> Iterator[str]:
    """Обучающие примеры целиком (карточка, вопрос, разбор, диагностики, ответ, объяснение)."""
    for rec in read_jsonl(SFT_DIR / f"{split}.jsonl", limit):
        yield sequence(rec).removesuffix("<EOS>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["sft"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for split in args.splits:
        t0 = time.time()
        out = TOKENS_DIR / f"{args.source}_{split}.bin"
        n = write_tokens(sft_texts(split, args.limit), out)
        print(f"{out.relative_to(ROOT)}: {n:,} tokens in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
