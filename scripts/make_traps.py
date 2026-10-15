"""Датасеты-ловушки для оценки отказов и режима «варианты» (этап 9, PLAN §8.4).

  .venv/Scripts/python.exe scripts/make_traps.py   # -> data/traps/*.csv

Механизмы порождения написаны отдельно от генераторов обучающих сценариев (datagen/scenarios), чтобы
ловушки не повторяли обучающие данные:
- loyalty_unidentifiable — участие в программе лояльности и траты зависят от неизмеряемого интереса
  к покупкам; инструмента и медиатора нет (вместе с data/demo/push_unidentifiable.csv);
- stores_all_treated, clinics_all_treated — панели, где все юниты получили воздействие одновременно:
  контрольной группы нет, общий сдвиг в тот же месяц от эффекта не отделить;
- newsletter_short_series — один ряд, 5 недель до и 5 после: для ITS слишком коротко;
- card_offer_poor_overlap, icu_poor_overlap — воздействие почти полностью определяется несколькими
  ковариатами (не одним порогом): сравнимых юнитов в группах почти нет.
Вопросы к ним — eval/trap_questions.jsonl, оценка — eval/traps.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "traps"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loyalty_unidentifiable(rng) -> pd.DataFrame:
    n = 3000
    interest = rng.normal(size=n)  # не измеряется
    age = rng.integers(18, 75, n)
    tenure = rng.integers(1, 120, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    joined = rng.random(n) < sigmoid(1.5 * interest + 0.01 * (age - 45))
    spend = 900 + 4 * (age - 45) + 2.5 * tenure + 160 * joined + 420 * interest + rng.normal(0, 150, n)
    return pd.DataFrame({"age": age, "tenure_months": tenure, "region": region,
                         "joined_loyalty": np.where(joined, "yes", "no"), "annual_spend": spend.round(2)})


def all_treated_panel(rng, units: list[str], unit_col: str, months: int, start: int, treat_col: str,
                      outcome_col: str, base: float, shock: float) -> pd.DataFrame:
    """Все юниты получают воздействие в месяц start; в тот же месяц у всех общий сдвиг (сезон, макро)."""
    dates = pd.period_range("2022-01", periods=months, freq="M").astype(str)
    level = rng.normal(base, 0.15 * base, len(units))
    rows = []
    for u, lv in zip(units, level):
        for t, d in enumerate(dates):
            post = t >= start
            y = lv + 0.004 * base * t + shock * post + 0.03 * base * np.sin(2 * np.pi * t / 12)
            rows.append({unit_col: u, "month": d, treat_col: int(post),
                         outcome_col: round(y + rng.normal(0, 0.02 * base), 2)})
    return pd.DataFrame(rows)


def newsletter_short_series(rng) -> pd.DataFrame:
    weeks = pd.date_range("2025-03-03", periods=10, freq="W-MON").strftime("%Y-%m-%d")
    after = np.arange(10) >= 5
    open_rate = 0.21 + 0.002 * np.arange(10) + 0.015 * after + rng.normal(0, 0.006, 10)
    return pd.DataFrame({"week": weeks, "new_design": after.astype(int), "open_rate": open_rate.round(4),
                         "emails_sent": rng.integers(18000, 22000, 10)})


def card_offer_poor_overlap(rng) -> pd.DataFrame:
    n = 4000
    z = rng.normal(size=(n, 3))
    credit_score = (680 + 60 * z[:, 0]).round()
    income = np.exp(10.8 + 0.4 * z[:, 1]).round(-2)
    debt_ratio = np.clip(0.35 - 0.12 * z[:, 2], 0.01, 0.95).round(3)
    offered = rng.random(n) < sigmoid(2.5 * z.sum(axis=1))
    spend = (3000 + 900 * z[:, 0] + 1200 * z[:, 1] + 500 * z[:, 2] + 400 * offered
             + rng.normal(0, 600, n))
    return pd.DataFrame({"credit_score": credit_score, "income": income, "debt_ratio": debt_ratio,
                         "age": rng.integers(21, 70, n), "premium_card_offer": offered.astype(int),
                         "card_spend": spend.round(2)})


def icu_poor_overlap(rng) -> pd.DataFrame:
    n = 2500
    z = rng.normal(size=(n, 3))
    severity = np.clip(50 + 15 * z[:, 0], 0, 100).round(1)
    age = np.clip(62 + 12 * z[:, 1], 18, 99).round()
    comorbidities = np.clip(np.round(2 + 1.3 * z[:, 2]), 0, 8)
    icu = rng.random(n) < sigmoid(2.8 * (z[:, 0] + 0.8 * z[:, 1] + 0.8 * z[:, 2]))
    stay = np.clip(6 + 3 * z[:, 0] + 1.5 * z[:, 1] + 1.2 * z[:, 2] - 1.0 * icu + rng.normal(0, 2, n), 1, None)
    return pd.DataFrame({"severity_score": severity, "age": age, "comorbidities": comorbidities,
                         "icu_admission": np.where(icu, "yes", "no"), "length_of_stay": stay.round(1)})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    data = {
        "loyalty_unidentifiable.csv": loyalty_unidentifiable(rng),
        "stores_all_treated.csv": all_treated_panel(rng, [f"store_{i:02d}" for i in range(1, 26)], "store", 24, 12,
                                                    "self_checkout", "revenue", 52000, 2600),
        "clinics_all_treated.csv": all_treated_panel(rng, [f"C{i:02d}" for i in range(1, 15)], "clinic", 30, 18,
                                                     "digital_triage", "avg_wait_minutes", 48, -6),
        "newsletter_short_series.csv": newsletter_short_series(rng),
        "card_offer_poor_overlap.csv": card_offer_poor_overlap(rng),
        "icu_poor_overlap.csv": icu_poor_overlap(rng),
    }
    for name, df in data.items():
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
