"""Детерминированные проверки допущений. Каждая возвращает DiagResult с цифрами и флагами.

Флаги — общий словарь с карточками методов (advisor.knowledge.FLAGS): правила и модель
опираются на них, а не на сырые числа.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from advisor.profiler import DataCard, TreatmentDesign, to_bool
from advisor.query import CausalQuery

# пороги вынесены сюда, чтобы их было видно и легко обсуждать
ALPHA = 0.05
SMALL_SAMPLE, LARGE_SAMPLE, HIGH_DIM = 200, 5000, 20
FEW_CONTROLS, SHORT_PRE, MANY_TREATED = 5, 5, 10
EXTREME_PS, POOR_OVERLAP_SHARE, GOOD_OVERLAP_SHARE = 0.05, 0.10, 0.02
RANDOM_AUC, BALANCED_SMD, IMBALANCED_SMD = 0.60, 0.10, 0.25
SC_GOOD_R2, SC_POOR_R2, OUTSIDE_HULL_SHARE = 0.90, 0.75, 0.50
HAC_MIN_PERIODS, MATERIAL_DRIFT = 12, 0.10
ITS_SHORT_PRE = 30
NONLIN_GAIN, NONLIN_SUBSAMPLE = 0.05, 3000
WEAK_IV_F = 10.0


@dataclass
class DiagResult:
    id: str
    applicable: bool
    values: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    message: str = ""


def _na(diag_id: str, why: str) -> DiagResult:
    return DiagResult(diag_id, applicable=False, message=why)


# ---------- размер выборки и устройство панели ----------

def series_split(df: pd.DataFrame, card: DataCard, query: CausalQuery) -> tuple[int, int] | None:
    """Для одиночного ряда: сколько периодов до и после начала воздействия."""
    if card.structure != "time_series" or not card.time_col or not query.treatment:
        return None
    if card.column(query.treatment).kind != "binary":
        return None
    d = to_bool(df[query.treatment]).values[np.argsort(df[card.time_col].astype(str).values)]
    if not d.any():
        return None
    first = int(np.argmax(d))
    return first, len(d) - first


def sample_size(df: pd.DataFrame, card: DataCard, query: CausalQuery,
                design: TreatmentDesign | None) -> DiagResult:
    flags, values = [], {"n_rows": card.n_rows}
    covs = covariate_columns(card, query)
    values["n_covariates"] = len(covs)
    if card.structure == "panel" and design is not None:
        values.update(n_treated_units=design.n_treated_units, n_control_units=design.n_control_units,
                      n_pre_periods=design.n_pre_periods, n_post_periods=design.n_post_periods)
        flags.append("policy_design")  # бинарное воздействие с временной структурой
        if design.n_control_units == 0:
            flags.append("no_controls")
        else:
            flags.append("has_controls")
            if design.n_control_units < FEW_CONTROLS:
                flags.append("few_controls")
        if design.n_pre_periods < SHORT_PRE:
            flags.append("short_pre_period")
        if design.n_treated_units == 1:
            flags.append("single_treated_unit")
        elif design.n_treated_units >= MANY_TREATED:
            flags.append("many_treated_units")
        if design.staggered:
            flags.append("staggered")
        msg = (f"{design.n_treated_units} treated and {design.n_control_units} control units, "
               f"{design.n_pre_periods} pre- and {design.n_post_periods} post-treatment periods")
    elif (split := series_split(df, card, query)) is not None:
        n_pre, n_post = split
        values.update(n_pre_periods=n_pre, n_post_periods=n_post)
        flags.append("no_controls")  # одиночный ряд: контрольной группы нет
        if n_pre < ITS_SHORT_PRE:
            flags.append("short_pre_period")
        msg = f"a single series with {n_pre} periods before and {n_post} after the intervention"
    else:
        if card.n_rows < SMALL_SAMPLE:
            flags.append("small_sample")
        elif card.n_rows >= LARGE_SAMPLE:
            flags.append("large_sample")
        msg = f"{card.n_rows} rows, {len(covs)} candidate covariates"
    if len(covs) >= HIGH_DIM:
        flags.append("high_dim")
    return DiagResult("sample_size", True, values, flags, msg)


# ---------- панели: параллельные тренды и качество синтетического контроля ----------

def _panel_frames(df: pd.DataFrame, card: DataCard, outcome: str, design: TreatmentDesign):
    """Матрица исхода (время × юнит) и маски treated / pre-period."""
    p = card.panel
    wide = df.pivot_table(index=p.time, columns=p.unit, values=outcome, aggfunc="mean").sort_index()
    treated_cols = [u for u in wide.columns if _same(u, design.treated_units)]
    control_cols = [u for u in wide.columns if u not in treated_cols]
    pre = wide.index < design.first_treated_time
    return wide, treated_cols, control_cols, pre


def _same(u, units) -> bool:
    return any(u == v or str(u) == str(v) for v in units)


def pretrend_test(df: pd.DataFrame, card: DataCard, query: CausalQuery,
                  design: TreatmentDesign | None) -> DiagResult:
    """Наклон разрыва «treated − среднее контрольных» до воздействия (OLS, HAC-ошибки).

    Значимый наклон = тренды до воздействия не параллельны.
    """
    if card.structure != "panel" or design is None:
        return _na("pretrend_test", "needs panel data with a treatment design")
    if design.n_control_units == 0:
        return _na("pretrend_test", "no control units to compare with")
    if design.n_pre_periods < 3:
        return _na("pretrend_test", "fewer than 3 pre-treatment periods")
    wide, tr, co, pre = _panel_frames(df, card, query.outcome, design)
    gap = (wide.loc[pre, tr].mean(axis=1) - wide.loc[pre, co].mean(axis=1)).dropna()
    t = np.arange(len(gap), dtype=float)
    X = sm.add_constant(t)
    if len(gap) < HAC_MIN_PERIODS:
        # на коротком ряду HAC-ошибки сильно занижены: обычный OLS с t-распределением
        fit = sm.OLS(gap.values, X).fit()
    else:
        maxlags = max(1, int(4 * (len(gap) / 100) ** (2 / 9)))
        fit = sm.OLS(gap.values, X).fit(cov_type="HAC", use_t=True,
                                        cov_kwds={"maxlags": maxlags, "use_correction": True})
    slope, p = float(fit.params[1]), float(fit.pvalues[1])
    drift = slope * (len(gap) - 1)
    sd = float(np.nanstd(wide.loc[pre].values))
    rel = abs(drift) / sd if sd > 0 else 0.0
    # «не параллельны» = значимо И заметно по величине (значимость одна ловит и микроскопический дрейф)
    violated = p < ALPHA and rel >= MATERIAL_DRIFT
    values = {"gap_slope": slope, "p_value": p, "drift_over_pre_period": drift,
              "drift_relative_to_sd": rel, "n_pre_periods": len(gap)}
    msg = (f"before treatment the gap between treated and control means drifts by {drift:+.3g} "
           f"({rel:.0%} of the outcome's SD, p={p:.3g}); trends are {'NOT ' if violated else ''}parallel")
    return DiagResult("pretrend_test", True, values, ["pretrend_violated" if violated else "pretrend_ok"], msg)


def sc_pre_fit(df: pd.DataFrame, card: DataCard, query: CausalQuery,
               design: TreatmentDesign | None) -> DiagResult:
    """Классический синтетический контроль (веса ≥ 0, сумма 1) на pre-периоде: насколько хорошо он
    воспроизводит траекторию treated и лежит ли treated внутри диапазона контрольных."""
    if card.structure != "panel" or design is None:
        return _na("sc_pre_fit", "needs panel data with a treatment design")
    if design.n_control_units < 2 or design.n_pre_periods < 3:
        return _na("sc_pre_fit", "needs at least 2 control units and 3 pre-treatment periods")
    wide, tr, co, pre = _panel_frames(df, card, query.outcome, design)
    y = wide.loc[pre, tr].mean(axis=1)
    Y = wide.loc[pre, co].dropna(axis=1)
    y = y.loc[Y.index]
    n = Y.shape[1]
    res = minimize(lambda w: np.sum((y.values - Y.values @ w) ** 2), np.full(n, 1 / n),
                   jac=lambda w: -2 * Y.values.T @ (y.values - Y.values @ w),
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 500})
    w = res.x
    fitted = Y.values @ w
    ss_res = float(np.sum((y.values - fitted) ** 2))
    ss_tot = float(np.sum((y.values - y.values.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    outside = float(np.mean((y.values > Y.values.max(axis=1)) | (y.values < Y.values.min(axis=1))))
    top = sorted(zip(Y.columns, w), key=lambda kv: -kv[1])[:5]
    flags = []
    if r2 >= SC_GOOD_R2:
        flags.append("sc_good_fit")
    elif r2 < SC_POOR_R2:
        flags.append("sc_poor_fit")
    if outside > OUTSIDE_HULL_SHARE:
        flags.append("treated_outside_hull")
    values = {"pre_r2": r2, "pre_rmspe": float(np.sqrt(ss_res / len(y))), "outside_hull_share": outside,
              "n_donors_used": int((w > 0.01).sum()), "top_weights": {str(k): round(float(v), 3) for k, v in top}}
    msg = (f"synthetic control reproduces the pre-treatment path with R²={r2:.2f} using "
           f"{values['n_donors_used']} donors; treated unit is outside the control range in {outside:.0%} of pre-periods")
    return DiagResult("sc_pre_fit", True, values, flags, msg)


# ---------- cross-section: пересечение групп, баланс, инструменты, разрыв ----------

def covariate_columns(card: DataCard, query: CausalQuery) -> list[str]:
    if query.covariates is not None:
        return list(query.covariates)
    # переменная порога — конфаундер (определяет воздействие и влияет на исход), её нужно учитывать
    skip = query.role_columns() - {query.running_variable}
    if card.panel:
        skip |= {card.panel.unit, card.panel.time}
        for d in card.designs:
            skip |= set(d.columns)
    if card.time_col:
        skip.add(card.time_col)
    return [c.name for c in card.columns
            if c.name not in skip and c.kind in ("binary", "integer", "numeric", "categorical")
            and c.missing_frac < 0.5 and not (c.kind == "categorical" and c.n_unique > 50)]


def _design_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = pd.get_dummies(df[cols], drop_first=True, dummy_na=False).astype(float)
    return X.fillna(X.median())


def _treatment_is_binary(card: DataCard, treatment: str) -> bool:
    return card.column(treatment).kind == "binary"


def overlap_balance(df: pd.DataFrame, card: DataCard, query: CausalQuery) -> DiagResult:
    """Propensity score (логрегрессия, кросс-валидированные предсказания) + стандартизованные разности."""
    if card.structure != "cross_section" or not query.treatment:
        return _na("overlap", "needs cross-sectional data and a treatment")
    if not _treatment_is_binary(card, query.treatment):
        return _na("overlap", "implemented for binary treatments")
    covs = covariate_columns(card, query)
    if not covs:
        return _na("overlap", "no covariates to check")
    d = to_bool(df[query.treatment]).astype(int).values
    X = _design_matrix(df, covs)
    m1, m0 = X[d == 1].mean(), X[d == 0].mean()
    pooled = np.sqrt((X[d == 1].var() + X[d == 0].var()) / 2).replace(0, np.nan)
    smd = ((m1 - m0) / pooled).fillna(0)
    folds = int(min(5, np.bincount(d).min()))
    if folds < 2:
        return _na("overlap", "too few units in one of the groups")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                               X.values, d, cv=folds, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(d, ps))
    extreme = float(np.mean((ps < EXTREME_PS) | (ps > 1 - EXTREME_PS)))
    max_smd = float(smd.abs().max())
    # при рандомизации SMD ~ N(0, 1/n1 + 1/n0): на малых выборках случайный дисбаланс велик,
    # поэтому пороги не ниже трёх стандартных ошибок
    smd_noise = 3 * float(np.sqrt(1 / max((d == 1).sum(), 1) + 1 / max((d == 0).sum(), 1)))
    flags = []
    if extreme > POOR_OVERLAP_SHARE:
        flags.append("poor_overlap")
    elif extreme < GOOD_OVERLAP_SHARE:
        flags.append("good_overlap")
    # баланс по наблюдаемым ковариатам не доказывает рандомизацию, если пользователь сам
    # сообщает о скрытом конфаундере или о том, что воздействие выбирали
    if (auc < RANDOM_AUC and max_smd < max(BALANCED_SMD, smd_noise)
            and not query.hidden_confounding and query.randomized is not False):
        flags.append("randomization_plausible")
    if max_smd > max(IMBALANCED_SMD, smd_noise):
        flags.append("imbalance")
    worst = smd.abs().sort_values(ascending=False).head(5)
    values = {"propensity_auc": auc, "extreme_propensity_share": extreme, "max_abs_smd": max_smd,
              "worst_smd": {k: round(float(smd[k]), 3) for k in worst.index}}
    msg = (f"covariates predict treatment with AUC={auc:.2f}; {extreme:.0%} of units have extreme "
           f"propensity scores; max standardized difference {max_smd:.2f}")
    return DiagResult("overlap", True, values, flags, msg)


def first_stage_strength(df: pd.DataFrame, card: DataCard, query: CausalQuery) -> DiagResult:
    """F-статистика инструмента в регрессии воздействия на инструмент и ковариаты."""
    if not (query.instrument and query.treatment):
        return _na("first_stage_strength", "no instrument specified")
    covs = [c for c in covariate_columns(card, query) if c != query.instrument]
    X = _design_matrix(df, [query.instrument] + covs)
    t = df[query.treatment]
    t = to_bool(t).astype(float) if card.column(query.treatment).kind == "binary" else t.astype(float)
    keep = t.notna()
    fit = sm.OLS(t[keep], sm.add_constant(X[keep])).fit(cov_type="HC1")
    inst_cols = [c for c in X.columns if c == query.instrument or c.startswith(query.instrument + "_")]
    f = float(fit.f_test(np.eye(len(fit.params))[[list(fit.params.index).index(c) for c in inst_cols]]).fvalue)
    flags = ["weak_instrument"] if f < WEAK_IV_F else []
    return DiagResult("first_stage_strength", True, {"first_stage_F": f}, flags,
                      f"first-stage F-statistic of the instrument is {f:.1f} (rule of thumb: weak if below 10)")


def find_running_variable(df: pd.DataFrame, card: DataCard, query: CausalQuery) -> DiagResult:
    """Ищет непрерывную переменную, порог которой определяет воздействие (кандидат для RDD)."""
    if card.structure != "cross_section" or not query.treatment or not _treatment_is_binary(card, query.treatment):
        return _na("running_variable", "needs cross-sectional data and a binary treatment")
    d = to_bool(df[query.treatment]).astype(int)
    candidates = [query.running_variable] if query.running_variable else [
        c.name for c in card.columns
        if c.kind in ("numeric", "integer") and c.n_unique >= 20 and c.name not in query.role_columns()]
    best = None
    for col in candidates:
        x = df[col]
        ok = x.notna()
        xs, ds = x[ok].values.astype(float), d[ok].values
        grid = np.unique(np.quantile(xs, np.linspace(0.05, 0.95, 91)))
        for c in grid:
            above = xs >= c
            acc = max(np.mean(above == ds), np.mean(above != ds))
            if best is None or acc > best[1]:
                best = (col, acc, c)
    if best is None:
        return _na("running_variable", "no continuous candidates")
    col, acc, cut = best
    xs, ds = df[col].values.astype(float), d.values
    h = 0.05 * (np.nanmax(xs) - np.nanmin(xs))
    left, right = ds[(xs >= cut - h) & (xs < cut)], ds[(xs >= cut) & (xs < cut + h)]
    jump = float(right.mean() - left.mean()) if len(left) and len(right) else 0.0
    flags = []
    if acc >= 0.85 and abs(jump) >= 0.5:
        flags.append("running_variable_found")
        if acc >= 0.99:
            flags.append("sharp_cutoff")
    values = {"variable": col, "cutoff": float(cut), "assignment_accuracy": float(acc), "jump_at_cutoff": jump}
    msg = (f"best threshold rule: {col} >= {cut:.4g} explains treatment for {acc:.0%} of units, "
           f"treatment probability jumps by {jump:+.2f} at the cutoff")
    return DiagResult("running_variable", True, values, flags, msg)


def nonlinearity(df: pd.DataFrame, card: DataCard, query: CausalQuery) -> DiagResult:
    """Насколько бустинг предсказывает исход лучше линейной модели (кросс-валидированный R²).
    Заметный выигрыш = связи нелинейны, линейная регрессия будет неверно специфицирована."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    if card.structure != "cross_section" or not (query.treatment and query.outcome):
        return _na("nonlinearity", "needs cross-sectional data with a treatment and an outcome")
    covs = covariate_columns(card, query)
    if not covs:
        return _na("nonlinearity", "no covariates")
    data = df.sample(n=min(len(df), NONLIN_SUBSAMPLE), random_state=0)
    kind = card.column(query.treatment).kind
    if kind == "binary":
        t = to_bool(data[query.treatment]).astype(float).rename("__t").to_frame()
    elif kind in ("numeric", "integer"):
        t = data[query.treatment].astype(float).rename("__t").to_frame()
    else:  # несколько вариантов воздействия — one-hot
        t = pd.get_dummies(data[query.treatment].astype(str), prefix="__t", drop_first=True).astype(float)
    X = pd.concat([_design_matrix(data, covs), t], axis=1)
    X = X.fillna(X.median()).fillna(0.0)
    y = data[query.outcome].astype(float).values
    r2_lin = float(np.mean(cross_val_score(LinearRegression(), X.values, y, cv=3, scoring="r2")))
    gbm = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.15)
    r2_gbm = float(np.mean(cross_val_score(gbm, X.values, y, cv=3, scoring="r2")))
    gain = r2_gbm - r2_lin
    flag = "nonlinear" if gain > NONLIN_GAIN else "linear_ok"
    return DiagResult("nonlinearity", True, {"r2_linear": r2_lin, "r2_boosting": r2_gbm, "gain": gain}, [flag],
                      f"boosting explains the outcome with R2={r2_gbm:.2f} vs {r2_lin:.2f} for a linear model")


def normality(df: pd.DataFrame, card: DataCard, query: CausalQuery) -> DiagResult:
    """Доля переменных с негауссовским шумом: тест Жарка–Бера по остаткам регрессии каждой переменной
    на остальные. Остатки ближе к шуму, чем сами переменные (сумма многих причин гауссеет по ЦПТ)."""
    cols = [c.name for c in card.columns if c.kind in ("numeric", "integer") and c.n_unique >= 10]
    if len(cols) < 2:
        return _na("normality", "fewer than 2 continuous variables")
    X = df[cols].astype(float)
    X = ((X - X.mean()) / X.std()).fillna(0.0).values
    pvals = {}
    for j, c in enumerate(cols):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(X)), others])
        resid = X[:, j] - A @ np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        pvals[c] = float(stats.jarque_bera(resid).pvalue)
    share = float(np.mean([p < 0.01 for p in pvals.values()]))
    flags = ["non_gaussian" if share >= 0.5 else "gaussian_data"]
    return DiagResult("normality", True, {"non_gaussian_share": share, "p_values": pvals}, flags,
                      f"{share:.0%} of variables have clearly non-Gaussian noise")


# ---------- запуск всего применимого ----------

def run_diagnostics(df: pd.DataFrame, card: DataCard, query: CausalQuery) -> dict[str, DiagResult]:
    design = card.design_for(query.treatment) if card.structure == "panel" else None
    results = [sample_size(df, card, query, design)]
    if query.question_type in ("effect", "heterogeneity"):
        if card.structure == "panel":
            results += [pretrend_test(df, card, query, design), sc_pre_fit(df, card, query, design)]
        elif card.structure == "cross_section":
            results += [overlap_balance(df, card, query), find_running_variable(df, card, query),
                        nonlinearity(df, card, query)]
            if query.instrument:
                results.append(first_stage_strength(df, card, query))
    if query.question_type == "discovery":
        results.append(normality(df, card, query))
    return {r.id: r for r in results}


def all_flags(diags: dict[str, DiagResult]) -> set[str]:
    return {f for r in diags.values() for f in r.flags}
