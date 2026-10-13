"""Маленькие синтетические датасеты для тестов этапа 1 (полноценные генераторы — этап 2)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def smoking() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "demo" / "smoking.csv")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def rct(rng) -> pd.DataFrame:
    n = 2000
    age = rng.normal(40, 10, n)
    income = rng.lognormal(10, 0.5, n)
    treated = rng.integers(0, 2, n)
    spend = 50 + 0.5 * age + 0.001 * income + 5 * treated + rng.normal(0, 10, n)
    return pd.DataFrame({"age": age, "income": income, "got_coupon": treated, "spend": spend})


@pytest.fixture
def observational(rng) -> pd.DataFrame:
    """Воздействие сильно зависит от ковариат: overlap плохой, рандомизацией не пахнет."""
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(3 * x1 + 2 * x2)))
    t = rng.binomial(1, p)
    y = 2 * t + 3 * x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "region": rng.choice(["north", "south", "east"], n),
                         "treatment": t, "outcome": y})


@pytest.fixture
def rdd(rng) -> pd.DataFrame:
    n = 1500
    score = rng.uniform(0, 100, n)
    scholarship = (score >= 60).astype(int)
    earnings = 20 + 0.3 * score + 5 * scholarship + rng.normal(0, 3, n)
    return pd.DataFrame({"test_score": score, "scholarship": scholarship, "earnings": earnings,
                         "age": rng.integers(18, 25, n)})


@pytest.fixture
def iv(rng) -> pd.DataFrame:
    n = 3000
    u = rng.normal(size=n)  # скрытый конфаундер
    z = rng.binomial(1, 0.5, n)
    t = 0.8 * z + u + rng.normal(size=n)
    y = 1.5 * t + 2 * u + rng.normal(size=n)
    return pd.DataFrame({"distance_low": z, "education_years": t, "wage": y, "age": rng.integers(20, 60, n)})


@pytest.fixture
def time_series(rng) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    after = np.arange(120) >= 90
    sales = 100 + 0.2 * np.arange(120) + 10 * after + rng.normal(0, 2, 120)
    return pd.DataFrame({"date": dates.astype(str), "campaign_live": after, "sales": sales})


@pytest.fixture
def staggered_panel(rng) -> pd.DataFrame:
    rows = []
    for store in range(30):
        start = {0: 2015, 1: 2015, 2: 2015, 3: 2018, 4: 2018}.get(store)  # две когорты
        for year in range(2010, 2022):
            d = start is not None and year >= start
            rows.append({"store_id": f"S{store:02d}", "year": year, "new_layout": int(d),
                         "revenue": 100 + store + 2 * (year - 2010) + 8 * d + rng.normal(0, 1)})
    return pd.DataFrame(rows)


@pytest.fixture
def parallel_panel(rng) -> pd.DataFrame:
    """Много treated, общий тренд у всех групп: DiD должен быть уместен."""
    rows = []
    for unit in range(40):
        g = unit < 20
        for t in range(1, 13):
            rows.append({"unit": unit, "month": t, "treated_group": g, "post": t >= 7,
                         "y": 10 + unit * 0.5 + 1.0 * t + 3 * (g and t >= 7) + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


@pytest.fixture
def discovery(rng) -> pd.DataFrame:
    n = 2000
    a = rng.uniform(-1, 1, n)
    b = 0.8 * a + rng.uniform(-1, 1, n)
    c = 0.5 * b + rng.uniform(-1, 1, n)
    d = 0.7 * a + 0.3 * c + rng.uniform(-1, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
