"""SFT с претрейном и с нуля: validation loss ответа по шагам, светлая и тёмная версии для README.

  .venv/Scripts/python.exe scripts/plot_sft.py runs/sft_pretrained.csv runs/sft_scratch.csv docs/img/sft_loss
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from scripts.plot_loss import THEMES  # noqa: E402


def read_val(path: Path) -> tuple[list[int], list[float]]:
    steps, val = [], []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["val_loss"]:
                steps.append(int(row["step"]))
                val.append(float(row["val_loss"]))
    return steps, val


def plot(runs: list[tuple[str, Path]], out: Path, theme: str, title: str) -> None:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for (label, log), color in zip(runs, (t["train"], t["val"])):
        steps, val = read_val(log)
        ax.plot(steps, val, color=color, lw=2, solid_capstyle="round", marker="o", markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2)
        # подпись у конца линии: цвет несёт линия, текст — нейтральными чернилами
        ax.annotate(f"{label} {val[-1]:.3f}", (steps[-1], val[-1]), xytext=(8, -3),
                    textcoords="offset points", color=t["ink2"], fontsize=9)
    ax.set_yscale("log")
    ax.set_yticks([0.04, 0.06, 0.1, 0.2, 0.3])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(500, 3000, 500))
    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.set_xlabel("optimizer step (32 examples each, 1 epoch)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("validation loss on the answer (log scale)", color=t["ink2"], fontsize=9)
    ax.set_xlim(0, 3500)
    ax.set_title(title, color=t["ink"], fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    suffix = "" if theme == "light" else "_dark"
    fig.savefig(out.with_name(out.name + suffix + ".png"), facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pretrained", type=Path)
    ap.add_argument("scratch", type=Path)
    ap.add_argument("out", type=Path, help="путь без расширения: получатся <out>.png и <out>_dark.png")
    ap.add_argument("--title", default="Fine-tuning on the task: from the pretrained model vs from scratch")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs = [("pretrained", args.pretrained), ("from scratch", args.scratch)]
    for theme in THEMES:
        plot(runs, args.out, theme, args.title)
    print(f"saved {args.out}.png and {args.out}_dark.png")


if __name__ == "__main__":
    main()
