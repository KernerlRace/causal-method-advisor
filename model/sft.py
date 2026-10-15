"""SFT (этап 8): пример задачи → токены, маска loss и мягкая метка для распределения по методам.

  <CARD>профиль<Q>вопрос<PARSE>разбор<DIAG>диагностики<ANS><MODE:m> <M:a> 1.00 <M:b> 0.40<EXPL>объяснение<EOS>

Loss считается только на том, что модель выдаёт сама: разбор вопроса вместе с закрывающим <DIAG>
(проход 1) и ответ с объяснением до <EOS> (проход 2). Профиль, вопрос и диагностики — вход.
Текст пользователя (профиль с именами колонок, вопрос, диагностики) кодируется без спецтокенов,
поэтому «<M:did>» в вопросе не станет служебным токеном.

Позиция перед первым методом учится мягкой меткой: вместо одного «правильного» метода — веса всех
подходящих методов из эталона сценария, нормированные в распределение. Тогда softmax в этой позиции —
оценка того, насколько подходит каждый метод, и её можно читать напрямую (model/decode.py).

Подготовка кэша токенов (~1 мин), дальше его читает model/train_sft.py:
  .venv/Scripts/python.exe -m model.sft --splits train val test
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.data import TOKENIZER, TOKENS_DIR
from model.tokenizer import BPETokenizer
from model.train_tokenizer import SFT_DIR, methods_text, read_jsonl

# порядок полей разбора — как в datagen.explanations.parse_fields; модель выдаёт их в этом порядке
FIELD_ORDER = ("target", "treatment", "outcome", "instrument", "mediator", "running", "randomized", "hidden")


def parse_text(fields: dict) -> str:
    """Разбор в каноническом порядке полей, что бы ни пришло (например, правка ролей в интерфейсе)."""
    parts = [f"<T:{fields['type']}>"]
    parts += [f"{k}=`{fields[k]}`" for k in FIELD_ORDER if fields.get(k) is not None]
    return " ".join(parts)


def method_tokens(tok: BPETokenizer) -> dict[str, int]:
    return {s[3:-1]: i for s, i in tok.special.items() if s.startswith("<M:")}


@dataclass
class Encoded:
    ids: list[int]
    target: list[bool]   # True — токен предсказывает модель (входит в loss)
    soft_pos: int        # позиция предсказания первого метода (-1 — методов нет)
    soft: dict[int, float]  # id токена метода → вероятность в мягкой метке


class Prompter:
    """Собирает последовательности токенов по сегментам — одинаково для обучения и для инференса."""

    def __init__(self, tok: BPETokenizer):
        self.tok = tok
        self.sp = {name: tok.token_id(f"<{name}>") for name in ("CARD", "Q", "PARSE", "DIAG", "ANS", "EXPL", "EOS", "PAD")}
        self.methods = method_tokens(tok)

    def text(self, s: str) -> list[int]:
        return self.tok.encode(s, allow_special=False)

    def pass1(self, card: str, question: str) -> list[int]:
        """Вход прохода 1: модель продолжает разбором вопроса и закрывает его токеном <DIAG>."""
        sp = self.sp
        return [sp["CARD"], *self.text(card), sp["Q"], *self.text(question), sp["PARSE"]]

    def pass2(self, card: str, question: str, parse: str, diag: str) -> list[int]:
        """Вход прохода 2: разбор + результаты диагностик; модель продолжает режимом ответа и методами."""
        sp = self.sp
        return [*self.pass1(card, question), *self.tok.encode(parse), sp["DIAG"], *self.text(diag), sp["ANS"]]

    def encode(self, rec: dict) -> Encoded:
        sp = self.sp
        ids = [sp["CARD"], *self.text(rec["card"]), sp["Q"], *self.text(rec["question"]), sp["PARSE"]]
        target = [False] * len(ids)

        def add(chunk: list[int], is_target: bool) -> None:
            ids.extend(chunk)
            target.extend([is_target] * len(chunk))

        add(self.tok.encode(rec["parse"]) + [sp["DIAG"]], True)
        add(self.text(rec["diag"]), False)
        add([sp["ANS"]], False)
        answer_start = len(ids)
        add(self.tok.encode(methods_text(rec)) + [sp["EXPL"]] + self.text(rec["explanation"]) + [sp["EOS"]], True)
        soft_pos, soft = -1, {}
        if rec["methods"]:
            first = next(i for i in range(answer_start, len(ids)) if ids[i] in self.methods.values())
            soft_pos = first - 1  # позиция, где предсказывается первый метод
            total = sum(rec["methods"].values())
            soft = {self.methods[m]: w / total for m, w in rec["methods"].items()}
        return Encoded(ids, target, soft_pos, soft)


# ---------- набор примеров на диске ----------

def build_cache(split: str, out: Path, tokenizer: Path = TOKENIZER, limit: int | None = None) -> int:
    tok = BPETokenizer.load(tokenizer)
    pr = Prompter(tok)
    method_ids = sorted(pr.methods.values())
    col = {t: j for j, t in enumerate(method_ids)}
    ids, target, offsets, soft_pos, soft = [], [], [0], [], []
    for rec in read_jsonl(SFT_DIR / f"{split}.jsonl", limit):
        e = pr.encode(rec)
        ids.extend(e.ids)
        target.extend(e.target)
        offsets.append(len(ids))
        soft_pos.append(e.soft_pos)
        row = np.zeros(len(method_ids), dtype=np.float32)
        for t, p in e.soft.items():
            row[col[t]] = p
        soft.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.asarray(ids, dtype=np.uint16), target=np.asarray(target, dtype=np.bool_),
             offsets=np.asarray(offsets, dtype=np.int64), soft_pos=np.asarray(soft_pos, dtype=np.int32),
             soft=np.stack(soft), method_ids=np.asarray(method_ids, dtype=np.int64))
    return len(offsets) - 1


class SFTData:
    """Примеры из кэша; батчи — из примеров похожей длины, чтобы меньше считать впустую на паддинге."""

    def __init__(self, path: Path, limit: int | None = None):
        z = np.load(path)
        self.ids, self.target, self.offsets = z["ids"], z["target"], z["offsets"]
        self.soft_pos, self.soft, self.method_ids = z["soft_pos"], z["soft"], z["method_ids"]
        self.n = len(self.offsets) - 1 if limit is None else min(limit, len(self.offsets) - 1)
        self.lengths = np.diff(self.offsets)[:self.n]

    def __len__(self) -> int:
        return self.n

    def batches(self, batch_size: int, rng: np.random.Generator, pool: int = 64):
        """Бесконечный поток батчей: перемешать, внутри пула из pool батчей отсортировать по длине,
        порезать и снова перемешать порядок батчей."""
        while True:
            order = rng.permutation(self.n)
            out = []
            for s in range(0, self.n, batch_size * pool):
                chunk = order[s:s + batch_size * pool]
                chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
                out += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
            for j in rng.permutation(len(out)):
                yield out[j]

    def collate(self, idx, pad_id: int, device) -> dict[str, torch.Tensor]:
        T = int(self.lengths[idx].max()) - 1
        x = np.full((len(idx), T), pad_id, dtype=np.int64)
        y = np.full((len(idx), T), -100, dtype=np.int64)
        for r, i in enumerate(idx):
            a, b = self.offsets[i], self.offsets[i + 1]
            seq = self.ids[a:b].astype(np.int64)
            tgt = np.where(self.target[a + 1:b], seq[1:], -100)
            x[r, :b - a - 1] = seq[:-1]
            y[r, :b - a - 1] = tgt
        batch = {"x": torch.from_numpy(x), "y": torch.from_numpy(y),
                 "soft_pos": torch.from_numpy(self.soft_pos[idx].astype(np.int64)),
                 "soft": torch.from_numpy(self.soft[idx])}
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def sft_loss(hidden: torch.Tensor, lm_head: torch.nn.Module, batch: dict, method_ids: torch.Tensor,
             soft_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Кросс-энтропия на токенах ответа; в позиции первого метода — с мягкой меткой вместо one-hot.
    Среднее по всем позициям с целью, мягкая позиция весит soft_weight обычных токенов.
    hidden — model.hidden(x): логиты считаются только там, где есть цель (~37% позиций)."""
    y = batch["y"].clone()
    rows = (batch["soft_pos"] >= 0).nonzero(as_tuple=True)[0]
    cols = batch["soft_pos"][rows]
    y[rows, cols] = -100  # эта позиция учится мягкой меткой
    sel = y != -100
    hard_sum = F.cross_entropy(lm_head(hidden[sel]).float(), y[sel], reduction="sum")
    n_hard = sel.sum()
    soft_sum = torch.zeros((), device=hidden.device)
    if len(rows):
        logp = F.log_softmax(lm_head(hidden[rows, cols]).float(), dim=-1)[:, method_ids]
        soft_sum = -(batch["soft"][rows] * logp).sum()
    n = n_hard + soft_weight * len(rows)
    loss = (hard_sum + soft_weight * soft_sum) / n
    stats = {"hard": (hard_sum / n_hard.clamp(min=1)).detach(),
             "soft": (soft_sum / max(len(rows), 1)).detach()}
    return loss, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    for split in args.splits:
        t0 = time.time()
        out = TOKENS_DIR / f"sft_examples_{split}.npz"
        n = build_cache(split, out, limit=args.limit)
        data = SFTData(out)
        L = data.lengths
        print(f"{out.name}: {n:,} examples, {L.sum():,} tokens, {data.target.sum():,} in the loss; "
              f"length median {int(np.median(L))}, max {L.max()}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
