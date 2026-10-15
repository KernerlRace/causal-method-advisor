"""SFT на задаче (этап 8): loss только на разборе и ответе, мягкая метка в позиции первого метода.

  .venv/Scripts/python.exe -m model.sft                                     # кэш токенов примеров
  # с претрейном (этап 7) и с нуля — для сравнения
  .venv/Scripts/python.exe -m model.train_sft --init checkpoints/pretrain_best.pt --run sft_pretrained
  .venv/Scripts/python.exe -m model.train_sft --init scratch --run sft_scratch --lr 1e-3
  # мало данных: на 5% примеров разница между претрейном и его отсутствием видна сильнее
  .venv/Scripts/python.exe -m model.train_sft --init checkpoints/pretrain_best.pt --limit 4500 --epochs 3 --run sft_pretrained_5pct

Валидация — по одним и тем же примерам при каждой оценке, с подстановкой правильного префикса
(teacher forcing): loss ответа, loss мягкой метки, точность типа вопроса, режима ответа и первого
метода (argmax по токенам методов среди лучших в эталоне). Лучший по val loss чекпойнт —
checkpoints/<run>_best.pt.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from model.data import TOKENIZER, TOKENS_DIR
from model.limits import GpuDuty, limit_cpu
from model.sft import SFTData, sft_loss
from model.tokenizer import BPETokenizer
from model.train_pretrain import autocast_for, lr_at
from model.transformer import GPT, PRESETS, GPTConfig, make_optimizer

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def validate(model: GPT, data: SFTData, n: int, batch_size: int, pad_id: int, groups: dict, device, amp,
             duty: GpuDuty | None = None) -> dict:
    """Метрики на первых n примерах val; groups — id токенов типа вопроса и режима ответа."""
    model.eval()
    method_ids = torch.as_tensor(data.method_ids, device=device)
    sums = {"loss": 0.0, "hard": 0.0, "soft": 0.0}
    hits = {"type": [0, 0], "mode": [0, 0], "method": [0, 0]}
    batches = 0
    for s in range(0, min(n, len(data)), batch_size):
        idx = np.arange(s, min(s + batch_size, n, len(data)))
        b = data.collate(idx, pad_id, device)
        with amp():
            h = model.hidden(b["x"])
            loss, st = sft_loss(h, model.lm_head, b, method_ids)
        sums["loss"] += loss.item()
        sums["hard"] += st["hard"].item()
        sums["soft"] += st["soft"].item()
        batches += 1
        for name, ids in groups.items():
            ids = torch.as_tensor(ids, device=device)
            where = torch.isin(b["y"], ids)
            pred = ids[model.lm_head(h[where])[:, ids].argmax(-1)]
            hits[name][0] += int((pred == b["y"][where]).sum())
            hits[name][1] += int(where.sum())
        rows = (b["soft_pos"] >= 0).nonzero(as_tuple=True)[0]
        if len(rows):
            scores = model.lm_head(h[rows, b["soft_pos"][rows]])[:, method_ids]
            soft = b["soft"][rows]
            best = soft >= soft.max(-1, keepdim=True).values - 1e-6
            hits["method"][0] += int(best[torch.arange(len(rows)), scores.argmax(-1)].sum())
            hits["method"][1] += len(rows)
        if duty is not None:
            duty.step()
    model.train()
    out = {k: v / batches for k, v in sums.items()}
    out.update({f"acc_{k}": h / max(t, 1) for k, (h, t) in hits.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="чекпойнт претрейна или 'scratch' — случайная инициализация")
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="base", help="только для --init scratch")
    ap.add_argument("--train", type=Path, default=TOKENS_DIR / "sft_examples_train.npz")
    ap.add_argument("--val", type=Path, default=TOKENS_DIR / "sft_examples_val.npz")
    ap.add_argument("--limit", type=int, default=None, help="взять только первые N обучающих примеров")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=None, help="по умолчанию lr/10")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--soft-weight", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-examples", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-duty", type=float, default=0.65, help="доля времени, когда GPU считает: потолок нагрузки (фон — ещё ~25%%)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    amp, amp_dtype = autocast_for(device)
    limit_cpu()
    duty = GpuDuty(device, args.gpu_duty)
    tok = BPETokenizer.load(TOKENIZER)
    torch.manual_seed(args.seed)
    if args.init == "scratch":
        cfg = GPTConfig(vocab_size=tok.vocab_size, dropout=args.dropout, attention="sdpa", **PRESETS[args.preset])
        model = GPT(cfg).to(device)
    else:
        model, _ = GPT.load(Path(args.init), device)
        if model.cfg.vocab_size != tok.vocab_size:
            raise SystemExit(f"{args.init}: словарь {model.cfg.vocab_size}, у токенизатора {tok.vocab_size}")
        model.cfg.attention, model.cfg.dropout = "sdpa", args.dropout
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = args.dropout
    model.train()

    train, val = SFTData(args.train, args.limit), SFTData(args.val)
    pad = tok.token_id("<PAD>")
    method_ids = torch.as_tensor(train.method_ids, device=device)
    groups = {"type": [i for s, i in tok.special.items() if s.startswith("<T:")],
              "mode": [i for s, i in tok.special.items() if s.startswith("<MODE:")]}
    per_step = args.batch * args.accum
    steps = int(np.ceil(args.epochs * len(train) / per_step))
    min_lr = args.min_lr if args.min_lr is not None else args.lr / 10
    opt = make_optimizer(model, args.lr, args.weight_decay)
    print(f"init {args.init} | {model.num_params() / 1e6:.1f}M params | {len(train):,} examples, "
          f"{per_step} per step, {steps} steps = {args.epochs:g} epochs | lr {args.lr:g} | {device} {amp_dtype or 'fp32'}",
          flush=True)

    ckpt = ROOT / "checkpoints" / f"{args.run}.pt"
    log_path = ROOT / "runs" / f"{args.run}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["step", "lr", "train_loss", "train_soft", "val_loss", "val_hard", "val_soft",
                  "val_acc_type", "val_acc_mode", "val_acc_method", "tokens_per_s", "elapsed_s"])

    rng = np.random.default_rng(args.seed)
    stream = train.batches(args.batch, rng)
    best = float("inf")
    t0 = t_last = time.time()
    run_loss = run_soft = 0.0
    n_since = tokens_since = 0
    for step in range(steps):
        lr = lr_at(step, args.lr, min_lr, args.warmup, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(args.accum):
            b = train.collate(next(stream), pad, device)
            with amp():
                loss, st = sft_loss(model.hidden(b["x"]), model.lm_head, b, method_ids, args.soft_weight)
            (loss / args.accum).backward()
            run_loss += loss.item() / args.accum
            run_soft += st["soft"].item() / args.accum
            tokens_since += b["x"].numel()
            duty.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        n_since += 1

        last = step == steps - 1
        do_eval = (step + 1) % args.eval_every == 0 or last
        if (step + 1) % args.log_every == 0 or do_eval:
            now = time.time()
            v = validate(model, val, args.val_examples, args.batch, pad, groups, device, amp, duty) if do_eval else None
            tps = tokens_since / (now - t_last)
            row = [step + 1, f"{lr:.3e}", f"{run_loss / n_since:.4f}", f"{run_soft / n_since:.4f}"]
            row += [f"{v[k]:.4f}" for k in ("loss", "hard", "soft", "acc_type", "acc_mode", "acc_method")] if v else [""] * 6
            log.writerow(row + [f"{tps:.0f}", f"{now - t0:.0f}"])
            log_file.flush()
            msg = f"step {step + 1:5d}/{steps} | loss {run_loss / n_since:.4f} soft {run_soft / n_since:.3f}"
            if v:
                msg += (f" | val {v['loss']:.4f} soft {v['soft']:.3f} type {v['acc_type']:.3f} "
                        f"mode {v['acc_mode']:.3f} method {v['acc_method']:.3f}")
            print(msg + f" | {tps / 1000:.1f}k tok/s | {(now - t0) / 60:.1f} min", flush=True)
            run_loss = run_soft = 0.0
            n_since = tokens_since = 0
            t_last = time.time()
            if v:
                if v["loss"] < best:
                    best = v["loss"]
                    model.save(ckpt.with_name(f"{args.run}_best.pt"), step=step + 1, val=v, args=vars(args) | {
                        "train": str(args.train), "val": str(args.val)})
                model.save(ckpt, step=step + 1, val=v)
    log_file.close()
    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
