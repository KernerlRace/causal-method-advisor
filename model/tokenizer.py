"""Byte-level BPE с нуля: обучение слияний, кодирование, декодирование.

Текст сначала режется регуляркой на куски (слово с ведущим пробелом, до трёх цифр, пунктуация,
пробелы), слияния идут только внутри куска. Базовый алфавит — 256 байт, поэтому закодировать
можно любую строку, а decode(encode(s)) == s.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from advisor.knowledge import load_cards

# [^\W\d_] — буква любого алфавита; подчёркивание идёт с пунктуацией, чтобы `recovery_days`
# резалось на recovery / _ / days и части имён колонок переиспользовались
PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+[\r\n]*"""
           r"""|\s*[\r\n]+|\s+(?!\S)|\s+""")

MARKUP = ["<PAD>", "<EOS>", "<CARD>", "<Q>", "<PARSE>", "<DIAG>", "<ANS>", "<EXPL>"]
QUESTION_TYPES = ["effect", "heterogeneity", "discovery", "not_causal"]
MODES = ["recommend", "options", "refuse"]


def default_special_tokens() -> list[str]:
    """Разметка, тип вопроса, режим ответа и по токену на каждый метод из карточек."""
    return (MARKUP + [f"<T:{t}>" for t in QUESTION_TYPES] + [f"<MODE:{m}>" for m in MODES]
            + [f"<M:{m}>" for m in sorted(load_cards())])


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: list[str], pattern: str = PATTERN):
        self.merges = [tuple(m) for m in merges]
        self.pattern = pattern
        self._split = re.compile(pattern)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        first_special = 256 + len(self.merges)
        self.special = {tok: first_special + i for i, tok in enumerate(special_tokens)}
        self.special_by_id = {i: tok for tok, i in self.special.items()}
        # длинные токены первыми, чтобы префикс не перехватил совпадение
        alts = sorted(self.special, key=len, reverse=True)
        self._special_split = re.compile("(" + "|".join(map(re.escape, alts)) + ")") if alts else None
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    def token_id(self, special: str) -> int:
        return self.special[special]

    # ---------- обучение ----------

    @classmethod
    def train(cls, texts: Iterable[tuple[str, int]], vocab_size: int,
              special_tokens: list[str] | None = None, pattern: str = PATTERN,
              min_frequency: int = 2) -> BPETokenizer:
        """texts — пары (текст, вес). Спецтокены в обучающем тексте вырезаются и в слияния не попадают."""
        special_tokens = default_special_tokens() if special_tokens is None else special_tokens
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size={vocab_size} меньше 256 байт + {len(special_tokens)} спецтокенов")
        split = re.compile(pattern)
        strip = re.compile("|".join(map(re.escape, special_tokens))) if special_tokens else None

        chunks: Counter[bytes] = Counter()
        for text, weight in texts:
            if strip is not None:
                text = strip.sub(" ", text)
            for piece in split.findall(text):
                chunks[piece.encode("utf-8")] += weight

        words = [list(b) for b in chunks]
        counts = list(chunks.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += counts[wi]
                where[pair].add(wi)

        # куча с ленивым удалением: запись верна, только если её счёт совпадает с текущим;
        # при равных счётах выигрывает меньшая пара — обучение детерминировано
        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg:
                continue
            if -neg < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: Counter[tuple[int, int]] = Counter()
            for wi in where.pop(pair, ()):
                ids = words[wi]
                merged = _merge(ids, pair, new_id)
                if len(merged) == len(ids):
                    continue
                c = counts[wi]
                for p in zip(ids, ids[1:]):
                    changed[p] -= c
                for p in zip(merged, merged[1:]):
                    changed[p] += c
                    where[p].add(wi)
                words[wi] = merged
            del pair_counts[pair]
            for p, delta in changed.items():
                if p == pair or delta == 0:
                    continue
                pair_counts[p] += delta
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                else:
                    heapq.heappush(heap, (-pair_counts[p], p))
        return cls(merges, special_tokens, pattern)

    # ---------- кодирование ----------

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """allow_special=False — для текста пользователя: «<M:did>» в вопросе останется обычными байтами."""
        if not allow_special or self._special_split is None:
            return self._encode_ordinary(text)
        out: list[int] = []
        for part in self._special_split.split(text):
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in self._split.findall(text):
            cached = self._cache.get(piece)
            if cached is None:
                cached = self._encode_piece(piece.encode("utf-8"))
                if len(self._cache) < 200_000:
                    self._cache[piece] = cached
            out.extend(cached)
        return out

    def _encode_piece(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = _merge(ids, pair, 256 + rank)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self.special_by_id:
                parts.append(self.special_by_id[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------- сохранение ----------

    def save(self, path: Path) -> None:
        payload = {"version": 1, "pattern": self.pattern, "merges": self.merges,
                   "special_tokens": list(self.special)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([tuple(m) for m in payload["merges"]], payload["special_tokens"], payload["pattern"])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i, n = 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
