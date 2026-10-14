"""Генерация из чекпойнта: свой промпт или начало примера из SFT-набора.

  # пример из test: промпт до <ANS> (карточка, вопрос, разбор, диагностики), модель дописывает ответ
  .venv/Scripts/python.exe -m model.sample checkpoints/stage6_sft_lm.pt --sft test:0
  # грубая проверка на n примерах: совпадает ли разбор вопроса и первый метод с эталоном
  .venv/Scripts/python.exe -m model.sample checkpoints/stage6_sft_lm.pt --n 200
  .venv/Scripts/python.exe -m model.sample checkpoints/stage6_sft_lm.pt --prompt "<CARD>rows=500"

Это ещё не советник: формат и loss только на ответе — этап 8, constrained decoding и вероятности
методов из softmax — тоже. Здесь проверяем, что модель вообще научилась писать текст задачи.
"""
from __future__ import annotations

import argparse
import itertools
import re
import time
from pathlib import Path

import torch

from model.data import TOKENIZER
from model.tokenizer import BPETokenizer
from model.train_tokenizer import SFT_DIR, read_jsonl, sequence
from model.transformer import GPT


class Sampler:
    def __init__(self, ckpt: Path, device: torch.device):
        self.tok = BPETokenizer.load(TOKENIZER)
        self.model, _ = GPT.load(ckpt, device)
        self.model.cfg.attention = "manual"  # своё внимание; на генерации с KV-кэшем скорость не важна
        self.model.eval()
        self.device = device

    def complete(self, prompt: str, max_new: int, stop: str | None = "<EOS>", temperature: float = 0.0,
                 top_k: int | None = None, generator: torch.Generator | None = None) -> str:
        ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
        stop_id = self.tok.token_id(stop) if stop else None
        out = self.model.generate(ids, max_new, temperature=temperature, top_k=top_k, eos_id=stop_id,
                                  generator=generator)
        return self.tok.decode(out[0, ids.shape[1]:].tolist())


def split_at(full: str, marker: str) -> tuple[str, str]:
    i = full.index(marker) + len(marker)
    return full[:i], full[i:]


def first_method(text: str) -> str | None:
    m = re.search(r"<M:([a-z_]+)>", text)
    return m.group(1) if m else None


def parse_fields(parse: str) -> tuple[str | None, str | None, str | None]:
    kind = re.search(r"<T:([a-z_]+)>", parse)
    treatment = re.search(r"treatment=`([^`]*)`", parse)
    outcome = re.search(r"outcome=`([^`]*)`", parse)
    return tuple(m.group(1) if m else None for m in (kind, treatment, outcome))


def evaluate(sampler: Sampler, n: int, split: str) -> None:
    """Разбор: промпт до <PARSE>, генерация до <DIAG>. Метод: промпт до <ANS> с настоящими диагностиками.
    Берём n примеров, равномерно разнесённых по файлу; отдельно — шаблоны вопросов, не виденные в обучении."""
    recs = list(read_jsonl(SFT_DIR / f"{split}.jsonl"))
    step = max(1, len(recs) // n)
    recs = recs[::step][:n]
    hits: dict[str, list[int]] = {k: [] for k in ("type", "treatment+outcome", "parse exact", "answer mode", "top-1 method")}
    heldout: list[bool] = []
    t0 = time.time()
    for rec in recs:
        full = sequence(rec)
        prompt, rest = split_at(full, "<PARSE>")
        gold = rest.split("<DIAG>")[0]
        pred = sampler.complete(prompt, 64, stop="<DIAG>").removesuffix("<DIAG>")
        (pt, ptr, pout), (gt, gtr, gout) = parse_fields(pred), parse_fields(gold)
        hits["type"].append(pt == gt)
        hits["treatment+outcome"].append((ptr, pout) == (gtr, gout))
        hits["parse exact"].append(pred == gold)
        prompt, rest = split_at(full, "<ANS>")
        answer = sampler.complete(prompt, 48, stop="<EXPL>")
        hits["answer mode"].append(answer.split()[:1] == rest.split()[:1])
        hits["top-1 method"].append(first_method(answer) == first_method(rest))
        heldout.append(rec["heldout_template"])
    h = [i for i, flag in enumerate(heldout) if flag]
    print(f"{split}: {len(recs)} examples ({len(h)} on unseen question templates), greedy, {time.time() - t0:.0f}s")
    print(f"  {'':20s} {'all':>7s} {'unseen':>7s}")
    for name, v in hits.items():
        unseen = f"{sum(v[i] for i in h) / len(h):7.1%}" if h else "      —"
        print(f"  {name:20s} {sum(v) / len(v):7.1%} {unseen}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--prompt")
    ap.add_argument("--sft", help="split:index — взять пример из data/sft")
    ap.add_argument("--cut", default="<ANS>", help="до какого маркера пример идёт в промпт")
    ap.add_argument("--n", type=int, default=0, help="проверить на n примерах из --split (больше размера — весь split)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = Sampler(args.ckpt, device)
    if args.n:
        evaluate(sampler, args.n, args.split)
        return

    gold = None
    if args.sft:
        split, index = args.sft.split(":")
        rec = next(itertools.islice(read_jsonl(SFT_DIR / f"{split}.jsonl"), int(index), None))
        prompt, gold = split_at(sequence(rec), args.cut)
    else:
        prompt = args.prompt or "<CARD>"
    gen = torch.Generator(device=device).manual_seed(args.seed)
    text = sampler.complete(prompt, args.max_new, temperature=args.temperature, top_k=args.top_k, generator=gen)
    print(prompt)
    print("\n========== model ==========\n" + text)
    if gold is not None:
        print("\n========== reference ==========\n" + gold)


if __name__ == "__main__":
    main()
