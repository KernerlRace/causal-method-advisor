"""Эталонные оценщики эффектов (свои реализации на numpy / statsmodels / sklearn / cvxpy).

Нужны, чтобы (1) проверять генераторы сценариев: правильный метод восстанавливает заложенный
эффект, а наивный — нет; (2) позже — считать эффект рекомендованным методом (бонус-этап).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from advisor.profiler import TreatmentDesign, to_bool


@dataclass
class Estimate:
    method: str
    value: float
    se: float | None = None
    details: dict = field(default_factory=dict)


def _num(s: pd.Series) -> np.ndarray:
    """Бинарные колонки -> 0/1, остальные -> float."""
    if s.dtype == bool or s.dropna().nunique() == 2:
        return to_bool(s).astype(float).values
    return s.astype(float).values


def _X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0))
    X = pd.get_dummies(df[cols], drop_first=True).astype(float)
    return X.fillna(X.median()).values


# ---------- эксперименты и back-door ----------

def diff_in_means(df, treatment, outcome) -> Estimate:
    d, y = _num(df[treatment]) == 1, df[outcome].astype(float).values
    v = y[d].mean() - y[~d].mean()
    se = np.sqrt(y[d].var(ddof=1) / d.sum() + y[~d].var(ddof=1) / (~d).sum())
    return Estimate("diff_in_means", float(v), float(se))


def ols_adjusted(df, treatment, outcome, covariates) -> Estimate:
    X = np.column_stack([np.ones(len(df)), _num(df[treatment]), _X(df, covariates)])
    fit = sm.OLS(df[outcome].astype(float).values, X).fit(cov_type="HC1")
    return Estimate("regression", float(fit.params[1]), float(fit.bse[1]))


def _propensity(X, d, flexible: bool):
    model = (HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1) if flexible
             else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
    return model.fit(X, d).predict_proba(X)[:, 1]


def ipw(df, treatment, outcome, covariates, clip=0.01) -> Estimate:
    d, y, X = _num(df[treatment]), df[outcome].astype(float).values, _X(df, covariates)
    ps = np.clip(_propensity(X, d, flexible=False), clip, 1 - clip)
    w1, w0 = d / ps, (1 - d) / (1 - ps)
    v = np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0)  # оценка Хайека
    return Estimate("ipw", float(v), details={"max_weight": float(max(w1.max(), w0.max()))})


def aipw(df, treatment, outcome, covariates, folds=5, seed=0) -> Estimate:
    """Дважды робастная оценка с кросс-фиттингом и бустингом в обеих моделях."""
    d, y, X = _num(df[treatment]), df[outcome].astype(float).values, _X(df, covariates)
    psi = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        ps = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
        ps = np.clip(ps, 0.01, 0.99)
        mu = {}
        for g in (0, 1):
            m = tr[d[tr] == g]
            mu[g] = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1).fit(X[m], y[m]).predict(X[te])
        psi[te] = (mu[1] - mu[0] + d[te] * (y[te] - mu[1]) / ps
                   - (1 - d[te]) * (y[te] - mu[0]) / (1 - ps))
    return Estimate("doubly_robust", float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi))))


def dml_plr(df, treatment, outcome, covariates, folds=5, seed=0) -> Estimate:
    """Частично линейная модель Y = θT + g(X) + e: остатки Y и T на X (бустинг, кросс-фиттинг)."""
    t, y, X = _num(df[treatment]), df[outcome].astype(float).values, _X(df, covariates)
    binary = set(np.unique(t)) <= {0.0, 1.0}
    rt, ry = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        if binary:
            t_hat = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
        else:
            t_hat = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1).fit(X[tr], t[tr]).predict(X[te])
        y_hat = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1).fit(X[tr], y[tr]).predict(X[te])
        rt[te], ry[te] = t[te] - t_hat, y[te] - y_hat
    theta = np.sum(rt * ry) / np.sum(rt ** 2)
    eps = ry - theta * rt
    se = np.sqrt(np.mean(rt ** 2 * eps ** 2) / np.mean(rt ** 2) ** 2 / len(y))
    return Estimate("dml", float(theta), float(se))


def t_learner_cate(df, treatment, outcome, covariates) -> np.ndarray:
    """CATE для каждого объекта: две модели исхода, разность предсказаний."""
    d, y, X = _num(df[treatment]), df[outcome].astype(float).values, _X(df, covariates)
    m1 = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1).fit(X[d == 1], y[d == 1])
    m0 = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1).fit(X[d == 0], y[d == 0])
    return m1.predict(X) - m0.predict(X)


# ---------- инструменты, front-door, разрыв ----------

def tsls(df, treatment, outcome, instrument, covariates=()) -> Estimate:
    """Двухшаговый МНК для одного инструмента, робастные (сэндвич) ошибки."""
    W = _X(df, list(covariates))
    one = np.ones((len(df), 1))
    X = np.column_stack([one, _num(df[treatment]), W])
    Z = np.column_stack([one, _num(df[instrument]), W])
    y = df[outcome].astype(float).values
    ZX_inv = np.linalg.inv(Z.T @ X)
    beta = ZX_inv @ Z.T @ y
    e = y - X @ beta
    V = ZX_inv @ (Z.T * e ** 2) @ Z @ ZX_inv.T
    return Estimate("iv", float(beta[1]), float(np.sqrt(V[1, 1])))


def front_door_linear(df, treatment, mediator, outcome) -> Estimate:
    """Линейный front-door: (эффект T на M) × (эффект M на Y при контроле T)."""
    t, m, y = _num(df[treatment]), df[mediator].astype(float).values, df[outcome].astype(float).values
    a = sm.OLS(m, sm.add_constant(t)).fit().params[1]
    b = sm.OLS(y, sm.add_constant(np.column_stack([m, t]))).fit().params[1]
    return Estimate("front_door", float(a * b), details={"t_on_m": float(a), "m_on_y": float(b)})


def rdd_local_linear(df, outcome, running, cutoff, treatment=None, bandwidth=None) -> Estimate:
    """Локально-линейная регрессия с треугольным ядром по обе стороны порога.
    Если задан treatment и разрыв нечёткий — оценка Вальда (fuzzy RDD)."""
    x = df[running].astype(float).values - cutoff
    if bandwidth is None:
        bandwidth = 1.5 * np.std(x) * len(x) ** (-1 / 5) * 4
    w = np.clip(1 - np.abs(x) / bandwidth, 0, None)
    keep = w > 0
    above = (x >= 0).astype(float)

    def jump(v):
        X = np.column_stack([np.ones(keep.sum()), above[keep], x[keep], above[keep] * x[keep]])
        return sm.WLS(v[keep], X, weights=w[keep]).fit()

    fy = jump(df[outcome].astype(float).values)
    if treatment is None:
        return Estimate("rdd", float(fy.params[1]), float(fy.bse[1]), {"bandwidth": bandwidth})
    ft = jump(_num(df[treatment]))
    return Estimate("rdd", float(fy.params[1] / ft.params[1]),
                    details={"bandwidth": bandwidth, "first_stage_jump": float(ft.params[1])})


# ---------- панели ----------

def _wide(df, unit, time, outcome) -> pd.DataFrame:
    return df.pivot_table(index=time, columns=unit, values=outcome, aggfunc="mean").sort_index()


def _split(wide, design: TreatmentDesign, start=None):
    start = design.first_treated_time if start is None else start
    treated = [u for u in wide.columns if any(u == v or str(u) == str(v) for v in design.treated_units)]
    controls = [u for u in wide.columns if u not in treated]
    pre = wide.index < start
    return treated, controls, pre


def did_twfe(df, unit, time, outcome, d) -> Estimate:
    """Двусторонние фиксированные эффекты (альтернирующее вычитание средних), кластеры по юнитам."""
    frame = pd.DataFrame({"u": df[unit].values, "t": df[time].values,
                          "y": df[outcome].astype(float).values, "d": np.asarray(d, dtype=float)}).dropna()
    for col in ("y", "d"):
        v = frame[col].copy()
        for _ in range(100):
            v = v - v.groupby(frame["u"]).transform("mean")
            v = v - v.groupby(frame["t"]).transform("mean")
        frame[col + "_dm"] = v
    fit = sm.OLS(frame["y_dm"], frame["d_dm"]).fit(cov_type="cluster",
                                                     cov_kwds={"groups": pd.factorize(frame["u"])[0]})
    return Estimate("did", float(fit.params.iloc[0]), float(fit.bse.iloc[0]))


def synthetic_control(df, unit, time, outcome, design: TreatmentDesign) -> Estimate:
    wide = _wide(df, unit, time, outcome)
    tr, co, pre = _split(wide, design)
    Y = wide[co].dropna(axis=1)
    y = wide[tr].mean(axis=1)
    Ypre, ypre = Y.loc[pre].values, y.loc[pre].values
    n = Ypre.shape[1]
    w = minimize(lambda w: np.sum((ypre - Ypre @ w) ** 2), np.full(n, 1 / n),
                 jac=lambda w: -2 * Ypre.T @ (ypre - Ypre @ w), bounds=[(0, 1)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP",
                 options={"maxiter": 1000, "ftol": 1e-12}).x
    synth = Y.values @ w
    att = float(np.mean(y.loc[~pre].values - synth[~pre]))
    return Estimate("synthetic_control", att, details={"weights": dict(zip(map(str, Y.columns), np.round(w, 4)))})


def _solve(problem: cp.Problem, var: cp.Variable) -> np.ndarray:
    """Решает QP; если солвер по умолчанию не справился — пробует другие."""
    for solver in (None, cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver) if solver else problem.solve()
        except (cp.error.SolverError, ValueError):
            continue
        if var.value is not None:
            return var.value
    raise RuntimeError("SDID weights: no solver converged")


def _sdid_block(wide: pd.DataFrame, treated: list, controls: list, pre: np.ndarray) -> float:
    """SDID для блочного дизайна (Arkhangelsky et al., 2021; реализация по Facure, гл. 25)."""
    Yco = wide[controls].dropna(axis=1)
    # веса не меняются при аффинном преобразовании исхода (есть intercept, ζ масштабируется вместе с Y),
    # поэтому нормируем ради численной устойчивости и возвращаем эффект в исходный масштаб
    m, s = float(np.nanmean(Yco.values[pre])), float(np.nanstd(Yco.values[pre])) or 1.0
    Ytr = (wide[treated].mean(axis=1).values - m) / s
    Ypre_co, Ypost_co = (Yco.values[pre] - m) / s, (Yco.values[~pre] - m) / s
    T_pre, N_co = Ypre_co.shape
    n_treated_post = len(treated) * (~pre).sum()
    zeta = n_treated_post ** 0.25 * np.nanstd(np.diff(Ypre_co, axis=0), ddof=1)

    # веса юнитов: тренд treated ≈ взвешенные контрольные + intercept, L2-штраф
    w = cp.Variable(N_co + 1)
    X = np.column_stack([np.ones(T_pre), Ypre_co])
    omega = np.clip(_solve(cp.Problem(
        cp.Minimize(cp.sum_squares(X @ w - Ytr[pre]) + T_pre * zeta ** 2 * cp.sum_squares(w[1:])),
        [cp.sum(w[1:]) == 1, w[1:] >= 0]), w)[1:], 0, None)

    # веса периодов: среднее post контрольных ≈ взвешенные pre-периоды + intercept
    lam = cp.Variable(T_pre + 1)
    Xt = np.vstack([np.ones(N_co), Ypre_co])
    lmb = np.clip(_solve(cp.Problem(
        cp.Minimize(cp.sum_squares(lam @ Xt - Ypost_co.mean(axis=0))),
        [cp.sum(lam[1:]) == 1, lam[1:] >= 0]), lam)[1:], 0, None)

    # взвешенная разность разностей по четырём ячейкам
    tr_post, tr_pre = Ytr[~pre].mean(), lmb @ Ytr[pre]
    co_post, co_pre = omega @ Ypost_co.mean(axis=0), omega @ (lmb @ Ypre_co)
    return float(((tr_post - tr_pre) - (co_post - co_pre)) * s)


def sdid(df, unit, time, outcome, design: TreatmentDesign) -> Estimate:
    """SDID; при поэтапном внедрении — по когортам (контроль = никогда не получавшие воздействие),
    ATT усредняется с весами по числу treated-наблюдений."""
    wide = _wide(df, unit, time, outcome)
    _, never, _ = _split(wide, design)
    effects, weights = {}, {}
    for start, units in design.cohorts.items():
        cols = [u for u in wide.columns if any(u == v or str(u) == str(v) for v in units)]
        pre = wide.index < start
        effects[start] = _sdid_block(wide, cols, never, pre)
        weights[start] = len(cols) * (~pre).sum()
    total = sum(weights.values())
    att = sum(effects[k] * weights[k] / total for k in effects)
    return Estimate("sdid", float(att), details={"cohort_effects": {str(k): v for k, v in effects.items()}})


def seasonal_period(times) -> int | None:
    """Период сезонности по шагу дат: день → 7, неделя → 52, месяц → 12; для чисел — неизвестен."""
    parsed = pd.to_datetime(pd.Series(times).astype(str), errors="coerce")
    if parsed.isna().mean() > 0.1:
        return None
    step = parsed.sort_values().diff().dt.days.median()
    return 7 if step <= 1.5 else 52 if step <= 8 else 12 if step <= 32 else None


def its_segmented(df, time, outcome, d, harmonics: int = 2) -> Estimate:
    """Сегментированная регрессия: тренд + сезонные гармоники + сдвиг уровня + изменение наклона;
    эффект — среднее отклонение от контрфактического тренда за post-период."""
    order = np.argsort(df[time].astype(str).values)
    y = df[outcome].astype(float).values[order]
    dd = np.asarray(d, dtype=float)[order]
    t = np.arange(len(y), dtype=float)
    t0 = t[dd == 1].min()
    since = np.where(dd == 1, t - t0, 0.0)
    cols = [t, dd, since]
    period = seasonal_period(df[time].values)
    if period:
        for k in range(1, harmonics + 1):
            cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    fit = sm.OLS(y, sm.add_constant(np.column_stack(cols))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    g = np.zeros(len(fit.params))
    g[2], g[3] = 1.0, since[dd == 1].mean()  # эффект = уровень + наклон × среднее время после
    effect = float(g @ fit.params)
    se = float(np.sqrt(g @ fit.cov_params() @ g))
    return Estimate("its", effect, se, details={"level_shift": float(fit.params[2]),
                                                    "slope_change": float(fit.params[3]),
                                                    "seasonal_period": period})


# ---------- поиск структуры ----------

def discover(df: pd.DataFrame, method: str, alpha: float = 0.01) -> np.ndarray:
    """Матрица смежности A[i, j] = 1, если найдено ребро i -> j (для ненаправленных — в обе стороны)."""
    data = df.astype(float).values
    data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)  # порог на веса не зависит от масштаба
    p = data.shape[1]
    if method == "lingam":
        from causallearn.search.FCMBased.lingam import DirectLiNGAM
        model = DirectLiNGAM()
        model.fit(data)
        B = model.adjacency_matrix_  # B[j, i] != 0: i -> j
        return (np.abs(B.T) > 0.05).astype(int)
    if method == "pc":
        from causallearn.search.ConstraintBased.PC import pc
        G = pc(data, alpha=alpha, show_progress=False).G.graph
    elif method == "ges":
        from causallearn.search.ScoreBased.GES import ges
        G = ges(data)["G"].graph
    else:
        raise ValueError(method)
    A = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            # causal-learn: G[j, i] == 1 и G[i, j] == -1 означает i -> j; -1/-1 — ненаправленное
            if G[j, i] == 1 and G[i, j] == -1:
                A[i, j] = 1
            elif G[i, j] == -1 and G[j, i] == -1:
                A[i, j] = A[j, i] = 1
    return A
