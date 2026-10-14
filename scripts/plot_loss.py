"""График обучения из runs/<run>.csv: train и validation loss по шагам, светлая и тёмная версии для README.

  .venv/Scripts/python.exe scripts/plot_loss.py runs/pretrain.csv docs/img/pretrain_loss --skip 250
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", train="#2a78d6", val="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", train="#3987e5", val="#d95926"),
}


def read_log(path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    steps, train, val_steps, val = [], [], [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_steps.append(int(row["step"]))
                val.append(float(row["val_loss"]))
    return steps, train, val_steps, val


def plot(log: Path, out: Path, theme: str, skip: int, title: str) -> None:
    t = THEMES[theme]
    steps, train, val_steps, val = read_log(log)
    keep = [i for i, s in enumerate(steps) if s > skip]
    steps, train = [steps[i] for i in keep], [train[i] for i in keep]
    keep = [i for i, s in enumerate(val_steps) if s > skip]
    val_steps, val = [val_steps[i] for i in keep], [val[i] for i in keep]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.plot(steps, train, color=t["train"], lw=2, solid_capstyle="round", label="train (with dropout)")
    ax.plot(val_steps, val, color=t["val"], lw=2, solid_capstyle="round", label="validation",
            marker="o", markersize=6, markeredgecolor=t["surface"], markeredgewidth=2)
    # подписи у концов линий: цвет несёт маркер, текст — нейтральными чернилами
    ax.annotate(f"validation {val[-1]:.2f}", (val_steps[-1], val[-1]), xytext=(8, 6),
                textcoords="offset points", color=t["ink2"], fontsize=9)
    ax.annotate(f"train {train[-1]:.2f}", (steps[-1], train[-1]), xytext=(8, -12),
                textcoords="offset points", color=t["ink2"], fontsize=9)

    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.set_xlabel("optimizer step (32,768 tokens each)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("cross-entropy loss", color=t["ink2"], fontsize=9)
    ax.set_xlim(right=steps[-1] * 1.13)
    ax.set_title(title, color=t["ink"], fontsize=11, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=t["ink2"])
    for line in leg.get_lines():
        line.set_linewidth(2)
    fig.tight_layout()
    suffix = "" if theme == "light" else "_dark"
    fig.savefig(out.with_name(out.name + suffix + ".png"), facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("out", type=Path, help="путь без расширения: получатся <out>.png и <out>_dark.png")
    ap.add_argument("--skip", type=int, default=0, help="не показывать первые шаги (там loss ~ ln(vocab))")
    ap.add_argument("--title", default="Pretraining, 34M parameters, 5.6 epochs of a 23M-token corpus")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        plot(args.log, args.out, theme, args.skip, args.title)
    print(f"saved {args.out}.png and {args.out}_dark.png")


if __name__ == "__main__":
    main()
