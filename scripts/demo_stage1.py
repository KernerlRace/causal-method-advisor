"""Демо этапа 1: CSV + вручную заданный запрос → профиль, диагностики, вердикты правил.

Пример (Prop 99):
  .venv/Scripts/python.exe scripts/demo_stage1.py data/demo/smoking.csv --treatment california --outcome cigsale
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advisor.diagnostics import run_diagnostics  # noqa: E402
from advisor.knowledge import load_cards  # noqa: E402
from advisor.profiler import load_csv, profile  # noqa: E402
from advisor.query import CausalQuery  # noqa: E402
from advisor.rules import FORBIDDEN, evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--type", default="effect", choices=["effect", "heterogeneity", "discovery", "not_causal"])
    ap.add_argument("--treatment")
    ap.add_argument("--outcome")
    ap.add_argument("--instrument")
    ap.add_argument("--mediator")
    ap.add_argument("--randomized", choices=["yes", "no"])
    ap.add_argument("--hidden-confounding", action="store_true")
    a = ap.parse_args()

    df = load_csv(a.csv)
    card = profile(df)
    query = CausalQuery(a.type, treatment=a.treatment, outcome=a.outcome, instrument=a.instrument,
                        mediator=a.mediator, randomized=None if a.randomized is None else a.randomized == "yes",
                        hidden_confounding=a.hidden_confounding or None)
    print("=== DATA CARD ===")
    print(card.to_text())

    diags = run_diagnostics(df, card, query)
    print("\n=== DIAGNOSTICS ===")
    for r in diags.values():
        if r.applicable:
            print(f"[{r.id}] {r.message}")
            print(f"    flags: {', '.join(r.flags) or '-'}")
        else:
            print(f"[{r.id}] n/a: {r.message}")

    cards = load_cards()
    print("\n=== RULES ===")
    for v in evaluate(card, query, diags, cards):
        if v.status == FORBIDDEN:
            continue
        print(f"{v.status.upper():8s} {cards[v.method].name}")
        for p in v.pros:
            print(f"    + {p}")
        for c in v.cons:
            print(f"    - {c}")
        for r in v.reasons:
            print(f"    ! {r}")
    forbidden = [v for v in evaluate(card, query, diags, cards) if v.status == FORBIDDEN]
    print(f"\nFORBIDDEN ({len(forbidden)}):")
    for v in forbidden:
        print(f"    {v.method}: {v.reasons[0]}")


if __name__ == "__main__":
    main()
