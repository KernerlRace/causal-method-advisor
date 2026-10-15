"""Диаграмма надёжности на test: уверенность модели в лучшем методе против доли верных ответов — до и после
температуры, подобранной на val, и сколько ответов при какой уверенности. Светлая и тёмная версии для README.
(Токен режима калиброван и без температуры — ECE 0.001, почти все ответы с уверенностью ~1: рисовать нечего.)

  .venv/Scripts/python.exe -m scripts.plot_calibration eval/results/calibration.json docs/img/calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from scripts.plot_loss import THEMES  # noqa: E402

MIN_BIN = 10  # корзины, где меньше примеров, не рисуем: доля верных там — шум


def panel(ax, t: dict, raw: dict, scaled: dict, title: str) -> None:
    ax.set_facecolor(t["surface"])
    ax.plot([0, 1], [0, 1], color=t["axis"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("perfect calibration", (0.3, 0.3), xytext=(10, -4), textcoords="offset points",
                color=t["muted"], fontsize=8)
    for res, color, label in ((raw, t["val"], "raw softmax"), (scaled, t["train"], f"temperature T={scaled['T']:.2f}")):
        pts = [p for p in res["points"] if p["n"] >= MIN_BIN]
        xs, ys = [p["conf"] for p in pts], [p["acc"] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, marker="o", markersize=6, markeredgecolor=t["surface"],
                markeredgewidth=2, zorder=3, label=f"{label} — ECE {res['ece']:.3f}")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_aspect("equal")
    ax.grid(color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    ax.set_xlabel("confidence", color=t["ink2"], fontsize=9)
    ax.set_title(title, color=t["ink"], fontsize=10, loc="left", pad=8)
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False, handlelength=1.6)
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def histogram(ax, t: dict, raw: dict, scaled: dict) -> None:
    """Сколько ответов в каждой корзине уверенности: два ряда рядом, тонкие столбики с зазором."""
    ax.set_facecolor(t["surface"])
    w = 1 / 15
    for k, (res, color) in enumerate(((raw, t["val"]), (scaled, t["train"]))):
        xs = [p["lo"] + w * (0.27 + 0.46 * k) for p in res["points"]]
        ax.bar(xs, [p["n"] for p in res["points"]], width=w * 0.42, color=color, edgecolor=t["surface"],
               linewidth=1, zorder=3)
    ax.set_yscale("log")
    top = max(p["n"] for res in (raw, scaled) for p in res["points"])
    ax.set_ylim(0.8, top * 1.6)
    ax.set_yticks([1, 10, 100, 1000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    ax.set_xlabel("confidence", color=t["ink2"], fontsize=9)
    ax.set_ylabel("test questions (log scale)", color=t["ink2"], fontsize=9)
    ax.set_title("How confident the answers are", color=t["ink"], fontsize=10, loc="left", pad=8)


def plot(report: dict, out: Path, theme: str, title: str) -> None:
    t = THEMES[theme]
    test = report["test"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(t["surface"])
    panel(axes[0], t, test["methods_raw"], test["methods_scaled"],
          f"Top method ({test['methods_raw']['n']:,} test questions)")
    histogram(axes[1], t, test["methods_raw"], test["methods_scaled"])
    axes[0].set_ylabel("share of correct answers", color=t["ink2"], fontsize=9)
    fig.suptitle(title, color=t["ink"], fontsize=11, x=0.06, ha="left")
    fig.tight_layout()
    suffix = "" if theme == "light" else "_dark"
    fig.savefig(out.with_name(out.name + suffix + ".png"), facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    ap.add_argument("out", type=Path, help="путь без расширения: получатся <out>.png и <out>_dark.png")
    ap.add_argument("--title", default="Confidence in the top method: test split, temperature fitted on validation")
    args = ap.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        plot(report, args.out, theme, args.title)
    print(f"saved {args.out}.png and {args.out}_dark.png")


if __name__ == "__main__":
    main()
