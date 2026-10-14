"""Обучение языковой модели на потоке токенов: претрейн (этап 7) и проверочные прогоны этапа 6.

  # претрейн на открытом корпусе (python -m model.data corpus)
  .venv/Scripts/python.exe -m model.train_pretrain --steps 6000 --run pretrain
  # модель переобучается на одном батче: loss → ~0
  .venv/Scripts/python.exe -m model.train_pretrain --overfit --steps 300 --run stage6_overfit
  # короткое обучение на текстах SFT: модель начинает писать текст в формате задачи
  .venv/Scripts/python.exe -m model.train_pretrain --train data/tokens/sft_train.bin --val data/tokens/sft_val.bin --steps 1200 --run stage6_sft_lm

На Ampere (3060 Ti) — bf16 autocast, где bf16 нет — fp16 с GradScaler. AdamW, warmup + cosine,
gradient accumulation, клиппинг градиента. Лог — runs/<run>.csv, чекпойнт с оптимизатором —
checkpoints/<run>.pt (сохраняется при каждой оценке, --resume продолжает с него), лучший по val —
checkpoints/<run>_best.pt: на нескольких эпохах конец обучения может уже переобучаться.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from model.data import TOKENIZER, TOKENS_DIR, TokenStream
from model.tokenizer import BPETokenizer
from model.transformer import GPT, PRESETS, GPTConfig, make_optimizer

ROOT = Path(__file__).resolve().parent.parent


def lr_at(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Линейный разгон warmup шагов, затем косинус от max_lr до min_lr к шагу total."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def autocast_for(device: torch.device):
    """bf16 там, где он есть (Ampere и новее), иначе fp16; на CPU — обычный fp32."""
    if device.type != "cuda":
        return contextlib.nullcontext, None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return (lambda: torch.autocast("cuda", dtype=dtype)), dtype


@torch.no_grad()
def evaluate(model: GPT, stream: TokenStream, n_batches: int, batch_size: int, device, amp) -> float:
    """Средний loss на одних и тех же окнах при каждой оценке — чтобы точки на графике были сравнимы."""
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = stream.batch(batch_size, model.cfg.context, rng, device)
        with amp():
            losses.append(model(x, y)[1].item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=TOKENS_DIR / "pretrain_train.bin")
    ap.add_argument("--val", type=Path, default=TOKENS_DIR / "pretrain_val.bin")
    ap.add_argument("--run", default="pretrain", help="имя прогона: runs/<run>.csv, checkpoints/<run>.pt")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="base")
    ap.add_argument("--context", type=int, default=None, help="по умолчанию — из пресета")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--attention", choices=["manual", "sdpa"], default="sdpa")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="шагов накопления градиента на один шаг оптимизатора")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--overfit", action="store_true", help="учиться на одном и том же батче: проверка, что модель может выучить что угодно")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp, amp_dtype = autocast_for(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    ckpt_path = ROOT / "checkpoints" / f"{args.run}.pt"
    log_path = ROOT / "runs" / f"{args.run}.csv"
    start_step, extra = 0, {}
    if args.resume and ckpt_path.exists():
        model, extra = GPT.load(ckpt_path, device)
        start_step = extra["step"]
        print(f"resume {ckpt_path.name} from step {start_step}")
    else:
        vocab = BPETokenizer.load(TOKENIZER).vocab_size
        overrides = {k: v for k, v in (("context", args.context),) if v is not None}
        cfg = GPTConfig(vocab_size=vocab, dropout=args.dropout, attention=args.attention,
                        **{**PRESETS[args.preset], **overrides})
        torch.manual_seed(args.seed)
        model = GPT(cfg).to(device)
    cfg = model.cfg
    model.train()
    opt = make_optimizer(model, args.lr, args.weight_decay)
    if "optimizer" in extra:
        opt.load_state_dict(extra["optimizer"])

    train = TokenStream(args.train)
    val = TokenStream(args.val) if args.val.exists() and not args.overfit else None
    rng = np.random.default_rng([args.seed, start_step])  # при --resume окна не повторяют уже пройденные
    fixed = train.batch(args.batch, cfg.context, rng, device) if args.overfit else None
    tokens_per_step = args.batch * cfg.context * args.accum
    print(f"model {model.num_params() / 1e6:.1f}M params | {cfg.n_layer}L d{cfg.d_model} h{cfg.n_head} "
          f"ctx{cfg.context} | {device} {amp_dtype or 'fp32'} | train {len(train):,} tokens, "
          f"{tokens_per_step:,} per step, {args.steps * tokens_per_step / len(train):.2f} epochs", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not (args.resume and log_path.exists())
    log_file = log_path.open("a" if not new_log else "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "lr", "train_loss", "val_loss", "grad_norm", "tokens_per_s", "elapsed_s"])

    best_val = extra.get("best_val", float("inf"))
    t_start = t_last = time.time()
    loss_sum, n_since = torch.zeros((), device=device), 0
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.lr, args.min_lr, args.warmup, args.steps)
        for group in opt.param_groups:
            group["lr"] = lr
        for _ in range(args.accum):
            x, y = fixed if fixed is not None else train.batch(args.batch, cfg.context, rng, device)
            with amp():
                _, loss = model(x, y)
            scaler.scale(loss / args.accum).backward()
            loss_sum += loss.detach() / args.accum
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        n_since += 1

        last = step == args.steps - 1
        do_eval = val is not None and ((step + 1) % args.eval_every == 0 or last)
        if (step + 1) % args.log_every == 0 or last or do_eval:
            now = time.time()
            train_loss = (loss_sum / n_since).item()
            tps = n_since * tokens_per_step / (now - t_last)
            val_loss = evaluate(model, val, args.eval_batches, args.batch, device, amp) if do_eval else None
            log.writerow([step + 1, f"{lr:.3e}", f"{train_loss:.4f}",
                          "" if val_loss is None else f"{val_loss:.4f}", f"{grad_norm.item():.3f}",
                          f"{tps:.0f}", f"{now - t_start:.0f}"])
            log_file.flush()
            print(f"step {step + 1:5d} | loss {train_loss:.4f}"
                  + ("" if val_loss is None else f" | val {val_loss:.4f}")
                  + f" | lr {lr:.2e} | grad {grad_norm.item():.2f} | {tps / 1000:.1f}k tok/s"
                  + f" | {(now - t_start) / 60:.1f} min", flush=True)
            loss_sum.zero_()
            n_since, t_last = 0, time.time()
        if do_eval or (last and not args.overfit):
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                model.save(ckpt_path.with_name(f"{args.run}_best.pt"), step=step + 1, val_loss=val_loss)
            model.save(ckpt_path, step=step + 1, optimizer=opt.state_dict(), best_val=best_val,
                       args=vars(args) | {"train": str(args.train), "val": str(args.val)})
    log_file.close()
    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
