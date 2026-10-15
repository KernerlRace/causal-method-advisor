"""Сценарий = датасет + правильный ответ (gold) + метаданные для генерации вопросов."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from advisor.knowledge import load_cards
from advisor.query import CausalQuery
from datagen.naming import Concept, Namer, realize


@dataclass
class Scenario:
    name: str
    df: pd.DataFrame
    query: CausalQuery
    gold: dict[str, float]  # допустимые методы с весами, 1.0 = лучший выбор
    answer_mode: str  # recommend | options | refuse
    true_effect: float | None
    estimand: str | None
    theme: str
    concepts: dict[str, str]  # колонка -> человеческая фраза (для вопросов)
    params: dict
    true_graph: dict | None = None  # для discovery: {"nodes": [...], "edges": [[from, to], ...]}
    extras: dict = field(default_factory=dict)  # для проверок, не сохраняется (истинный CATE и т.п.)

    def meta(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("df", "extras")}
        d["n_rows"], d["n_cols"] = self.df.shape
        return _jsonable(d)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def answer_mode(gold: dict[str, float], refuse: bool = False) -> str:
    if refuse or not gold:
        return "refuse"
    return "options" if sum(w >= 0.9 for w in gold.values()) >= 2 else "recommend"


# сценарии с наблюдаемыми конфаундерами (и без порога): здесь слабое пересечение групп меняет ответ.
# В RDD пересечения нет по построению, и это не мешает: эффект оценивается у порога.
BACKDOOR = ("backdoor_small", "backdoor_large", "backdoor_targeted")
LARGE_BACKDOOR_ROWS = 3000  # с этого размера back-door сценарий устроен как backdoor_large (DML / DR)


def backdoor_gold(n_rows: int, k: int) -> dict[str, float]:
    """Эталон back-door при нормальном пересечении: мало строк — регрессия, много — DML и DR
    (нелинейность и линейность уточняет reconcile_gold)."""
    if n_rows >= LARGE_BACKDOOR_ROWS:
        return {"dml": 1.0, "doubly_robust": 1.0, "causal_forest": 0.6, "ipw": 0.3, "psm": 0.3, "regression": 0.2}
    return {"regression": 1.0, "psm": 0.6, "ipw": 0.5, "matching": 0.6 if k <= 4 else 0.3, "doubly_robust": 0.4}


def overlap_gold(n_rows: int, k: int, nonlinear: bool) -> dict[str, float]:
    """Эталон при флаге poor_overlap: больше POOR_OVERLAP_SHARE (10%) объектов с propensity score
    ближе EXTREME_PS (0.05) к 0 или 1 (advisor/diagnostics.py).

    Для таких объектов в другой группе нет сравнимых, и эффект «для всех» оценивается только
    экстраполяцией модели. Честный ответ — варианты: какой эффект оценивать, решает пользователь.
    Дважды робастная оценка после обрезки по propensity score — эффект там, где группы пересекаются;
    мэтчинг с калипером — эффект для тех, кому нашлась пара. Регрессия ниже: она молча экстраполирует;
    IPW последним: экстремальные веса. Если методу не хватает строк (min_rows из карточки), в варианты
    поднимается следующий, чтобы их всегда было два.
    """
    g = prune_gold({"doubly_robust": 1.0, "psm": 0.9, "matching": 0.6 if k <= 4 else 0.3,
                    "regression": 0.3 if nonlinear else 0.5, "ipw": 0.2}, n_rows)
    ranked = sorted(g, key=g.get, reverse=True)
    if ranked:
        g[ranked[0]] = 1.0
    if len(ranked) > 1:
        g[ranked[1]] = max(g[ranked[1]], 0.9)
    return g


def reconcile_gold(name: str, gold: dict[str, float], flags: set[str], params: dict) -> dict[str, float]:
    """Согласует эталон с тем, что реально показали диагностики.

    Модель видит не механизм порождения данных, а профиль и диагностики. Если тест случайно не увидел
    заложенное (или увидел то, чего нет), аналитик с теми же цифрами выбрал бы другой метод — и эталон
    должен совпадать с его выбором, иначе обучающие примеры противоречат своему же входу.
    """
    g = dict(gold)
    if name == "backdoor_targeted":  # отбор оказался мягким — как back-door сценарий того же размера
        if "poor_overlap" not in flags:
            g = backdoor_gold(params["n"], params["k"])
        name = "backdoor_large" if params["n"] >= LARGE_BACKDOOR_ROWS else "backdoor_small"
    if name == "panel_parallel" and "pretrend_violated" in flags:
        g = {"sdid": 1.0, "did": 0.5, "synthetic_control": 0.2}
    if name == "panel_factor":
        if "pretrend_ok" in flags:
            g["did"] = 0.6
        if "sc_good_fit" in flags and "treated_outside_hull" not in flags:
            g["synthetic_control"] = 0.9
        elif "sc_poor_fit" in flags or "treated_outside_hull" in flags:
            g["synthetic_control"] = 0.4
    if name == "rct" and "imbalance" in flags:
        g = {"regression": 1.0, "diff_in_means": 0.8}  # случайный дисбаланс → корректировка ковариатами
    if name == "continuous_treatment":
        if "nonlinear" in flags:
            g = {"dml": 1.0, "causal_forest": 0.6, "regression": 0.3}
        elif "linear_ok" in flags:
            g = {"regression": 1.0, "dml": 0.8}
    if name == "backdoor_large" and "linear_ok" in flags:
        g["regression"] = 0.6
    if name == "backdoor_small" and "nonlinear" in flags:
        g = {"regression": 0.6, "doubly_robust": 1.0, "psm": 0.6, "ipw": 0.5}
    if name in BACKDOOR and "poor_overlap" in flags:
        g = overlap_gold(params["n"], params["k"], "nonlinear" in flags)
    if name.startswith("discovery") and name != "discovery_hidden":
        if "gaussian_data" in flags:
            g = {"pc": 1.0, "ges": 1.0, "notears": 0.6, "fci": 0.6, "lingam": 0.1}
        elif "non_gaussian" in flags:
            g = {"lingam": 1.0, "pc": 0.6, "ges": 0.5, "notears": 0.5, "fci": 0.5}
    return g


def prune_gold(gold: dict[str, float], n_rows: int) -> dict[str, float]:
    """Убирает методы, которым заведомо не хватит строк (жёсткий min_rows из карточки)."""
    cards = load_cards()
    return {m: w for m, w in gold.items() if cards[m].requires.get("min_rows", 0) <= n_rows}


# ---------- кирпичики генерации ----------

def loguniform_int(rng, lo: int, hi: int) -> int:
    return int(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def effect_size(rng, allow_zero: bool = True) -> float:
    """Эффект в стандартных отклонениях исхода; иногда ровно ноль (нулевой эффект тоже бывает)."""
    if allow_zero and rng.random() < 0.12:
        return 0.0
    return float(rng.uniform(0.1, 0.8) * rng.choice([-1, 1]))


def covariates(namer: Namer, rng, n: int, k: int, exclude_cat: bool = False):
    """k ковариат из темы. Возвращает (dict колонок, матрица F эффективных стандартизованных признаков).

    F считается по наблюдаемым значениям (а не по латентным), иначе категориальные колонки
    теряли бы часть сигнала и появлялся бы остаточный конфаундинг, ломающий gold.
    """
    pool = tuple(c for c in namer.theme.covariates if not (exclude_cat and c.kind == "cat"))
    concepts = namer.sample(pool, k)
    cov = np.full((k, k), rng.uniform(0, 0.4))
    np.fill_diagonal(cov, 1.0)
    Z = rng.multivariate_normal(np.zeros(k), cov, size=n) if k else np.zeros((n, 0))
    cols, F = {}, np.zeros((n, len(concepts)))
    for j, c in enumerate(concepts):
        values = realize(c, Z[:, j], rng)
        cols[namer.covariate(c)] = values
        F[:, j] = effective(values)
    return cols, F


def effective(values) -> np.ndarray:
    v = np.asarray(values)
    if v.dtype.kind in "OUS":  # категории -> порядковый код
        codes = pd.Series(v).astype("category").cat.codes.values.astype(float)
        v = codes
    v = v.astype(float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0


def outcome_factor(concept: Concept, y_std: np.ndarray) -> float:
    """Множитель «стандартные единицы модели → единицы исхода»: SD исхода = масштаб понятия."""
    sd = float(np.std(y_std))
    return concept.scale / sd if sd > 0 else concept.scale


def positive(concept: Concept, values: np.ndarray) -> np.ndarray:
    """Положительная по смыслу величина ушла в минус (тяжёлый хвост) → сдвиг всей колонки на константу.
    Сдвиг не меняет ни разностей, ни наклонов, поэтому истинные эффекты остаются прежними."""
    values = np.asarray(values, dtype=float)
    if concept.loc > 0 and values.min() < 0:
        values = values - values.min() + 0.02 * concept.scale
    return values


def scale_outcome(concept: Concept, y_std: np.ndarray) -> np.ndarray:
    """Линейное преобразование (структура и эффекты сохраняются, эффект умножается на outcome_factor)."""
    return positive(concept, concept.loc + outcome_factor(concept, y_std) * (y_std - np.mean(y_std)))


def encode_binary(rng, d: np.ndarray):
    """Разные кодировки бинарного воздействия, как в реальных CSV."""
    style = rng.choice(["int", "int", "bool", "yesno", "label"])
    d = np.asarray(d).astype(bool)
    if style == "int":
        return d.astype(int)
    if style == "bool":
        return d
    if style == "yesno":
        return np.where(d, "yes", "no")
    return np.where(d, "treated", "control")


# ---------- время ----------

def time_axis(rng, kind: str, T: int):
    """Значения времени для T периодов и имя-основа колонки."""
    if kind == "year":
        start = int(rng.integers(1960, 2026 - T))
        return np.arange(start, start + T), "year"
    if kind == "month":
        start = pd.Timestamp(year=int(rng.integers(2005, 2020)), month=int(rng.integers(1, 13)), day=1)
        return pd.date_range(start, periods=T, freq="MS").strftime("%Y-%m").values, "month"
    if kind == "week":
        start = pd.Timestamp("2015-01-05") + pd.Timedelta(weeks=int(rng.integers(0, 400)))
        return pd.date_range(start, periods=T, freq="W-MON").strftime("%Y-%m-%d").values, "week"
    if kind == "date":
        start = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
        return pd.date_range(start, periods=T, freq="D").strftime("%Y-%m-%d").values, "date"
    return np.arange(1, T + 1), "period"


TIME_NAMES = {"year": ("year", "yr"), "month": ("month", "period"), "week": ("week", "week_start"),
              "date": ("date", "day"), "period": ("period", "t", "time")}
TIME_NAMES_RU = {"year": ("год",), "month": ("месяц",), "week": ("неделя",), "date": ("дата",), "period": ("период",)}


def time_name(namer: Namer, kind: str) -> str:
    if namer.anonymous:
        pool = ("t", "time", "period")
    elif namer.russian:
        pool = TIME_NAMES_RU[kind]
    else:
        pool = TIME_NAMES[kind]
    name = namer._unique(str(namer.rng.choice(pool)))
    namer.concepts[name] = kind if kind != "date" else "day"
    return name


def unit_labels(rng, namer: Namer, n_units: int):
    style = rng.choice(["int", "code", "name"])
    if style == "int" or namer.anonymous:
        return np.arange(1, n_units + 1)
    prefix = namer.theme.unit.names[0].split("_")[0][:1].upper()
    if style == "code":
        return np.array([f"{prefix}{i:03d}" for i in range(1, n_units + 1)])
    return np.array([f"{namer.theme.unit.phrase.replace(' ', '_')}_{i}" for i in range(1, n_units + 1)])


# ---------- финальная «порча» датасета ----------

def finalize(df: pd.DataFrame, namer: Namer, rng, protected: set[str], cross_section: bool) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    # шумовые колонки, не связанные ни с чем
    for c in namer.sample(namer.theme.covariates, int(rng.integers(0, 3))):
        z = rng.normal(size=n)
        df[namer.covariate(c)] = realize(c, z, rng)
    # ID объекта в cross-section
    if cross_section and rng.random() < 0.3:
        name = namer.pick(namer.theme.person, ("id", "row_id", "obs"), "id")
        df.insert(0, name, rng.permutation(n) + 1000)
    # пропуски в 1–3 ковариатах
    if rng.random() < 0.25:
        candidates = [c for c in df.columns if c not in protected and df[c].dtype.kind in "fi"]
        for c in rng.choice(candidates, size=min(len(candidates), int(rng.integers(1, 4))), replace=False) if candidates else []:
            mask = rng.random(n) < rng.uniform(0.01, 0.08)
            df[c] = df[c].astype(float)
            df.loc[mask, c] = np.nan
    # округление и порядок
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(int(rng.integers(2, 4)))
    cols = list(df.columns)
    if rng.random() < 0.7:
        rng.shuffle(cols)
    df = df[cols]
    if cross_section:
        df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    return df
