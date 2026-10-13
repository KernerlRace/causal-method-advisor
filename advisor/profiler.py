"""Профайлер CSV: типы колонок, структура данных (cross-section / panel / time series),
кандидаты на роли и устройство воздействия в панели.

Результат — DataCard: его читают диагностики и правила, а to_text() кормит модель.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

# подсказки ищем по целым словам имени: «week_start» — время, «education_years» и «days_since» — нет
TIME_TOKENS = {"year", "yr", "date", "time", "month", "week", "day", "quarter", "period", "t", "dt", "ds",
               "timestamp", "год", "дата", "время", "месяц", "неделя", "день", "квартал", "период"}
UNIT_HINTS = re.compile(
    r"(^id$|_id$|^id_|state|country|region|city|store|shop|user|client|customer|unit|firm|company|"
    r"school|district|county|hospital|штат|стран|регион|город|магазин|клиент|пользоват|компан)",
    re.I)
DATE_LIKE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})")
TRUE_WORDS = {"true", "yes", "y", "t", "1", "treated", "treatment", "да"}
FALSE_WORDS = {"false", "no", "n", "f", "0", "control", "нет"}


@dataclass
class ColumnProfile:
    name: str
    kind: str  # binary | integer | numeric | categorical | datetime | text | constant
    n_unique: int
    missing_frac: float
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    top_values: list[str] = field(default_factory=list)


@dataclass
class PanelInfo:
    unit: str
    time: str
    n_units: int
    n_periods: int
    balanced: bool


@dataclass
class TreatmentDesign:
    """Как устроено воздействие в панели: флаг группы × флаг периода или готовый индикатор."""
    kind: str  # group_post | indicator
    columns: tuple[str, ...]
    treated_units: list
    n_treated_units: int
    n_control_units: int
    cohorts: dict  # время начала воздействия -> список юнитов
    first_treated_time: object
    n_pre_periods: int
    n_post_periods: int
    staggered: bool
    reversible: bool


@dataclass
class DataCard:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    structure: str  # cross_section | panel | time_series
    panel: PanelInfo | None = None
    time_col: str | None = None
    designs: list[TreatmentDesign] = field(default_factory=list)

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def names(self, *kinds: str) -> list[str]:
        return [c.name for c in self.columns if c.kind in kinds]

    def design_for(self, treatment: str | None) -> TreatmentDesign | None:
        """Дизайн панели, в котором участвует колонка воздействия (или первый, если не указана)."""
        if not self.designs:
            return None
        if treatment is None:
            return self.designs[0]
        for d in self.designs:
            if treatment in d.columns:
                return d
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Компактная сериализация для модели (одна строка на факт)."""
        lines = [f"rows={self.n_rows} cols={self.n_cols} structure={self.structure}"]
        if self.panel:
            p = self.panel
            lines.append(f"panel unit={p.unit}({p.n_units}) time={p.time}({p.n_periods}) "
                         f"{'balanced' if p.balanced else 'unbalanced'}")
        if self.time_col:
            lines.append(f"time_series time={self.time_col}")
        for c in self.columns:
            s = f"col {c.name} {c.kind} uniq={c.n_unique}"
            if c.min is not None:
                s += f" range=[{_fmt(c.min)}..{_fmt(c.max)}]"
            if c.top_values:
                s += " values=" + "|".join(c.top_values[:5])
            if c.missing_frac > 0:
                s += f" missing={c.missing_frac:.0%}"
            lines.append(s)
        for d in self.designs:
            lines.append(
                f"design {d.kind}({','.join(d.columns)}) treated={d.n_treated_units} "
                f"controls={d.n_control_units} pre={d.n_pre_periods} post={d.n_post_periods} "
                f"{'staggered' if d.staggered else 'block'}{' reversible' if d.reversible else ''}")
        return "\n".join(lines)


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def to_bool(s: pd.Series) -> pd.Series:
    """Бинарная колонка -> bool. Для непонятных значений «True» — более редкое из двух."""
    if s.dtype == bool:
        return s
    vals = s.dropna().unique()
    if set(pd.Series(vals).astype(str).str.lower()) <= TRUE_WORDS | FALSE_WORDS:
        return s.astype(str).str.lower().isin(TRUE_WORDS)
    if pd.api.types.is_numeric_dtype(s) and set(vals) <= {0, 1}:
        return s == 1
    rare = s.value_counts().idxmin()
    return s == rare


def _looks_like_dates(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(50)
    if sample.empty or not sample.str.match(DATE_LIKE).mean() > 0.9:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.9


def profile_column(s: pd.Series) -> ColumnProfile:
    nonnull = s.dropna()
    n_unique = int(nonnull.nunique())
    prof = ColumnProfile(name=str(s.name), kind="constant", n_unique=n_unique,
                         missing_frac=float(s.isna().mean()))
    if n_unique <= 1:
        return prof
    if pd.api.types.is_datetime64_any_dtype(s) or (
            not pd.api.types.is_numeric_dtype(s) and s.dtype != bool and _looks_like_dates(s)):
        prof.kind = "datetime"
        dt = pd.to_datetime(nonnull, errors="coerce")
        prof.top_values = [str(dt.min().date()), str(dt.max().date())]
        return prof
    if s.dtype == bool or n_unique == 2:
        prof.kind = "binary"
        prof.top_values = [str(v) for v in nonnull.value_counts().index[:2]]
        if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
            prof.min, prof.max = float(nonnull.min()), float(nonnull.max())
        return prof
    if pd.api.types.is_numeric_dtype(s):
        x = nonnull.astype(float)
        prof.kind = "integer" if np.all(np.mod(x, 1) == 0) else "numeric"
        prof.min, prof.max = float(x.min()), float(x.max())
        prof.mean, prof.std = float(x.mean()), float(x.std())
        return prof
    avg_len = nonnull.astype(str).str.len().mean()
    prof.kind = "text" if avg_len > 30 and n_unique > 0.5 * len(nonnull) else "categorical"
    prof.top_values = [str(v) for v in nonnull.value_counts().index[:5]]
    return prof


def name_tokens(name: str) -> list[str]:
    """Слова имени колонки: snake_case, camelCase, пробелы, дефисы."""
    parts = re.split(r"[_\s\-.]+|(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])", name)
    return [p.lower() for p in parts if p]


def _time_score(df: pd.DataFrame, c: ColumnProfile) -> float:
    # нецелые числа временем не бывают (это длительности, доли, суммы)
    if c.kind not in ("integer", "datetime") or c.n_unique < 3:
        return 0.0
    score = 0.0
    if c.kind == "datetime":
        score += 3
    if TIME_TOKENS & set(name_tokens(c.name)):
        score += 2
    if c.kind == "integer":
        if 1900 <= c.min and c.max <= 2100:
            score += 2
        steps = np.diff(np.sort(df[c.name].dropna().unique()))
        if len(steps) and np.all(steps == steps[0]):
            score += 1
    return score


def _unit_score(c: ColumnProfile, n_rows: int) -> float:
    if c.kind not in ("integer", "categorical") or c.n_unique < 2 or c.n_unique > n_rows / 2:
        return 0.0
    score = 1.0 if c.kind == "categorical" else 0.5
    if UNIT_HINTS.search(c.name):
        score += 2
    return score


def _detect_structure(df: pd.DataFrame, cols: list[ColumnProfile]):
    n = len(df)
    time_cands = sorted(((s, c.name) for c in cols if (s := _time_score(df, c)) >= 2), reverse=True)
    unit_cands = sorted(((s, c.name) for c in cols if (s := _unit_score(c, n)) > 0), reverse=True)
    best_panel, ts_col = None, None
    for ts, t in time_cands:
        if df[t].nunique() == n and ts_col is None:
            ts_col = t
        if n / df[t].nunique() < 1.5:
            continue  # почти каждый период встречается один раз — это временной ряд, а не панель
        for us, u in unit_cands:
            if u == t or df.duplicated([u, t]).any():
                continue
            per_unit = df.groupby(u)[t].count()
            if (per_unit >= 2).mean() < 0.8:
                continue
            if best_panel is None or ts + us > best_panel[0]:
                best_panel = (ts + us, u, t)
    if best_panel:
        _, u, t = best_panel
        n_units, n_periods = df[u].nunique(), df[t].nunique()
        return "panel", PanelInfo(u, t, int(n_units), int(n_periods), n_units * n_periods == n), None
    if ts_col:
        return "time_series", None, ts_col
    return "cross_section", None, None


def _design_stats(df: pd.DataFrame, unit: str, time: str, d: pd.Series, kind: str,
                  columns: tuple[str, ...]) -> TreatmentDesign | None:
    if d.nunique() < 2:
        return None
    frame = pd.DataFrame({"u": df[unit], "t": df[time], "d": d.values}).sort_values(["u", "t"])
    treated = frame.loc[frame["d"], "u"].unique()
    first = frame[frame["d"]].groupby("u")["t"].min()
    cohorts: dict = {}
    for u, t0 in first.items():
        cohorts.setdefault(_py(t0), []).append(_py(u))
    # reversible: юнит выходит из воздействия после начала
    after_start = frame.merge(first.rename("t0"), left_on="u", right_index=True)
    reversible = bool((~after_start.loc[after_start["t"] >= after_start["t0"], "d"]).any())
    t_first = first.min()
    periods = np.sort(frame["t"].unique())
    # воздействие с первого периода или «мигающее» у многих объектов — это шум, а не дизайн
    switches = frame.groupby("u")["d"].apply(lambda s: int((s.astype(int).diff().abs() > 0).sum()))
    if (periods < t_first).sum() == 0 or (switches > 1).mean() > 0.3:
        return None
    return TreatmentDesign(
        kind=kind, columns=columns, treated_units=[_py(u) for u in treated],
        n_treated_units=int(len(treated)), n_control_units=int(frame["u"].nunique() - len(treated)),
        cohorts=dict(sorted(cohorts.items())), first_treated_time=_py(t_first),
        n_pre_periods=int((periods < t_first).sum()), n_post_periods=int((periods >= t_first).sum()),
        staggered=len(cohorts) > 1, reversible=reversible)


def _py(x):
    return x.item() if hasattr(x, "item") else x


def _detect_designs(df: pd.DataFrame, panel: PanelInfo, cols: list[ColumnProfile]) -> list[TreatmentDesign]:
    u, t = panel.unit, panel.time
    unit_level, time_level, indicators = [], [], []
    for c in cols:
        if c.kind != "binary" or c.name in (u, t):
            continue
        b = to_bool(df[c.name].fillna(False))
        within_unit = (b.groupby(df[u]).nunique() > 1).any()
        within_time = (b.groupby(df[t]).nunique() > 1).any()
        if within_time and not within_unit:
            unit_level.append(c.name)
        elif within_unit and not within_time:
            time_level.append(c.name)
        elif within_unit and within_time:
            indicators.append(c.name)
    designs = []
    for g in unit_level:
        for p in time_level:
            d = to_bool(df[g].fillna(False)) & to_bool(df[p].fillna(False))
            if (des := _design_stats(df, u, t, d, "group_post", (g, p))) is not None:
                designs.append(des)
    for col in indicators:
        if (des := _design_stats(df, u, t, to_bool(df[col].fillna(False)), "indicator", (col,))) is not None:
            designs.append(des)
    return designs


def profile(df: pd.DataFrame) -> DataCard:
    cols = [profile_column(df[c]) for c in df.columns]
    structure, panel, time_col = _detect_structure(df, cols)
    designs = _detect_designs(df, panel, cols) if panel else []
    return DataCard(n_rows=len(df), n_cols=df.shape[1], columns=cols, structure=structure,
                    panel=panel, time_col=time_col, designs=designs)


def load_csv(path) -> pd.DataFrame:
    """Читает CSV; разделитель (, ; таб) определяется автоматически."""
    return pd.read_csv(path, sep=None, engine="python")
