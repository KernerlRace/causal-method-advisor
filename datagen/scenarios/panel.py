"""Сценарии с временем: DiD (параллельные тренды), SC/SDID (факторная модель, тренды не параллельны),
поэтапное внедрение и одиночный временной ряд (ITS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.query import CausalQuery
from datagen.naming import Namer
from datagen.scenario import (Scenario, answer_mode, effect_size, encode_binary, finalize, positive, time_axis,
                              time_name, unit_labels)

POST_NAMES = ("post", "after", "after_treatment", "post_period", "after_launch", "is_post")
POST_NAMES_RU = ("после",)


def _bin_no_labels(rng, d):
    """Флаги периода кодируем без меток treated/control (они бессмысленны для времени)."""
    style = rng.choice(["int", "bool", "yesno"])
    d = np.asarray(d).astype(bool)
    return {"int": d.astype(int), "bool": d, "yesno": np.where(d, "yes", "no")}[style]


def _skeleton(namer: Namer, rng, n_units: int, T: int):
    kind = str(rng.choice(namer.theme.times))
    times, kind = time_axis(rng, kind, T)
    units = unit_labels(rng, namer, n_units)
    unit_col = namer.pick(namer.theme.unit, ("unit", "id", "unit_id"), "unit")
    time_col = time_name(namer, kind)
    return units, times, unit_col, time_col, kind


def _long(units, times, Y, unit_col, time_col, y_col):
    n_units, T = len(units), len(times)
    return pd.DataFrame({unit_col: np.repeat(units, T), time_col: np.tile(times, n_units),
                         y_col: Y.reshape(-1)})


def _treatment_columns(namer, rng, df, unit_col, time_col, treated_mask_units, start_idx_per_unit,
                       times, staggered: bool):
    """Добавляет колонки воздействия: пара «группа × после» или один индикатор D_it."""
    tc = namer.sample(namer.theme.treatments)[0]
    t_idx = np.tile(np.arange(len(times)), len(treated_mask_units))
    start = np.repeat(start_idx_per_unit, len(times))
    d = t_idx >= start  # start = inf для контрольных
    if not staggered and rng.random() < 0.5:
        g = namer.pick(tc, ("treated", "group", "treat"), "treatment")
        post_pool = POST_NAMES_RU if namer.russian else POST_NAMES
        p = namer._unique(str(rng.choice(post_pool)))
        namer.concepts[p] = "post-treatment period"
        df[g] = encode_binary(rng, np.repeat(treated_mask_units, len(times)))
        df[p] = _bin_no_labels(rng, t_idx >= start_idx_per_unit[treated_mask_units].min())
        return g, d, (g, p)
    col = namer.pick(tc, ("D", "treated", "treatment"), "treatment")
    df[col] = _bin_no_labels(rng, d)
    return col, d, (col,)


def _finish(name, namer, rng, df, query, gold, true_effect, params, roles) -> Scenario:
    df = finalize(df, namer, rng, protected=set(roles), cross_section=False)
    return Scenario(name, df, query, gold, answer_mode(gold), true_effect, "ATT", namer.theme.name,
                    dict(namer.concepts), params)


def panel_parallel(rng) -> Scenario:
    """Много treated, общий тренд (уровни групп различаются) → классический DiD."""
    namer = Namer(rng)
    n_units, T = int(rng.integers(20, 150)), int(rng.integers(6, 30))
    t0 = int(rng.integers(3, T - 1))
    treated = np.zeros(n_units, bool)
    treated[rng.choice(n_units, size=max(5, int(n_units * rng.uniform(0.2, 0.5))), replace=False)] = True
    alpha = rng.normal(size=n_units) + treated * rng.uniform(1, 2) * rng.choice([-1, 1])
    beta_t = np.linspace(0, rng.uniform(-1.5, 1.5), T) + np.cumsum(rng.normal(0, 0.15, T))
    tau = effect_size(rng)
    starts = np.where(treated, t0, np.inf)
    D = np.arange(T)[None, :] >= starts[:, None]
    noise = rng.uniform(0.2, 0.6)
    yc = namer.sample(namer.theme.outcomes)[0]
    Y_std = alpha[:, None] + beta_t[None, :] + tau * D + noise * rng.normal(size=(n_units, T))
    units, times, uc, tcol, kind = _skeleton(namer, rng, n_units, T)
    y = namer.pick(yc, ("Y", "y", "outcome"), "outcome")
    df = _long(units, times, positive(yc, yc.loc + yc.scale * Y_std), uc, tcol, y)
    tr_col, _, roles = _treatment_columns(namer, rng, df, uc, tcol, treated, starts, times, staggered=False)
    query = CausalQuery("effect", treatment=tr_col, outcome=y)
    gold = {"did": 1.0, "sdid": 0.8, "synthetic_control": 0.2}
    return _finish("panel_parallel", namer, rng, df, query, gold, tau * yc.scale,
                   {"n_units": n_units, "T": T, "t0": t0, "n_treated": int(treated.sum()), "time_kind": kind},
                   roles + (uc, tcol, y))


def panel_factor(rng) -> Scenario:
    """1–3 treated, собранные из контрольных с сильным трендом → тренды не параллельны → SDID / SC."""
    namer = Namer(rng)
    n_co, n_tr = int(rng.integers(12, 60)), int(rng.choice([1, 1, 2, 3]))
    T = int(rng.integers(15, 45))
    post = int(rng.integers(3, min(12, T - 10) + 1))
    t0 = T - post
    f = np.column_stack([np.linspace(-1, 1, T), np.cumsum(rng.normal(0, 0.3, T)),
                         np.sin(np.arange(T) * 2 * np.pi / rng.choice([4, 6, 12]))])
    f[:, 1] = (f[:, 1] - f[:, 1].mean()) / (f[:, 1].std() + 1e-9)
    amp = np.array([rng.uniform(1.0, 2.0), rng.uniform(0.3, 0.8), rng.uniform(0, 0.4)])
    lam_co, alpha_co = rng.normal(size=(n_co, 3)), rng.normal(size=n_co)
    level_shift = rng.random() < 0.3
    top = np.argsort(lam_co[:, 0] * rng.choice([-1, 1]))[-max(3, n_co // 3):]  # контрольные с крайним трендом
    lam_tr, alpha_tr = [], []
    for _ in range(n_tr):
        S = rng.choice(top, size=int(rng.integers(2, 5)), replace=False)
        w = rng.dirichlet(np.ones(len(S)))
        lam_tr.append(w @ lam_co[S] + rng.normal(0, 0.05, 3))
        alpha_tr.append(w @ alpha_co[S] + (rng.uniform(2.5, 4) * rng.choice([-1, 1]) if level_shift else 0))
    lam = np.vstack([lam_co, lam_tr])
    alpha = np.concatenate([alpha_co, alpha_tr])
    n_units = n_co + n_tr
    treated = np.arange(n_units) >= n_co
    tau = effect_size(rng, allow_zero=False)
    ramp = rng.random() < 0.4
    eff_t = np.where(np.arange(T) >= t0, tau * (1 + (0.5 * (np.arange(T) - t0) / post if ramp else 0)), 0)
    Y_std = alpha[:, None] + (lam * amp) @ f.T + treated[:, None] * eff_t[None, :] \
        + rng.uniform(0.05, 0.2) * rng.normal(size=(n_units, T))
    order = rng.permutation(n_units)  # treated не в конце списка
    Y_std, treated = Y_std[order], treated[order]
    starts = np.where(treated, t0, np.inf)
    yc = namer.sample(namer.theme.outcomes)[0]
    units, times, uc, tcol, kind = _skeleton(namer, rng, n_units, T)
    y = namer.pick(yc, ("Y", "y", "outcome"), "outcome")
    df = _long(units, times, positive(yc, yc.loc + yc.scale * Y_std), uc, tcol, y)
    tr_col, _, roles = _treatment_columns(namer, rng, df, uc, tcol, treated, starts, times, staggered=False)
    query = CausalQuery("effect", treatment=tr_col, outcome=y)
    gold = ({"sdid": 1.0, "synthetic_control": 0.4, "did": 0.1} if level_shift
            else {"sdid": 1.0, "synthetic_control": 0.9, "did": 0.1})
    true_att = float(eff_t[t0:].mean() * yc.scale)
    return _finish("panel_factor", namer, rng, df, query, gold, true_att,
                   {"n_controls": n_co, "n_treated": n_tr, "T": T, "post": post, "level_shift": level_shift,
                    "ramp": ramp, "time_kind": kind}, roles + (uc, tcol, y))


def panel_staggered(rng) -> Scenario:
    """Несколько когорт с разным временем начала и растущим эффектом → наивный TWFE смещён."""
    namer = Namer(rng)
    n_units, T = int(rng.integers(30, 120)), int(rng.integers(10, 30))
    n_cohorts = int(rng.integers(2, 4))
    starts_pool = np.sort(rng.choice(np.arange(3, T - 1), size=n_cohorts, replace=False))
    # мало «никогда не получавших» и быстро растущий эффект — именно тогда наивный TWFE сравнивает
    # поздние когорты с уже получившими воздействие и смещается
    n_treated = min(int(n_units * rng.uniform(0.6, 0.85)), n_units - 6)
    starts = np.full(n_units, np.inf)
    idx = rng.choice(n_units, size=n_treated, replace=False)
    starts[idx] = rng.choice(starts_pool, size=n_treated)
    alpha = rng.normal(size=n_units)
    beta_t = np.linspace(0, rng.uniform(-1.5, 1.5), T) + np.cumsum(rng.normal(0, 0.1, T))
    tau0 = effect_size(rng, allow_zero=False)
    tau1 = rng.uniform(0.15, 0.35) * np.sign(tau0)
    tt = np.arange(T)[None, :]
    D = tt >= starts[:, None]
    eff = np.where(D, tau0 + tau1 * (tt - np.where(np.isinf(starts), 0, starts)[:, None]), 0.0)
    Y_std = alpha[:, None] + beta_t[None, :] + eff + rng.uniform(0.2, 0.5) * rng.normal(size=(n_units, T))
    yc = namer.sample(namer.theme.outcomes)[0]
    units, times, uc, tcol, kind = _skeleton(namer, rng, n_units, T)
    y = namer.pick(yc, ("Y", "y", "outcome"), "outcome")
    df = _long(units, times, positive(yc, yc.loc + yc.scale * Y_std), uc, tcol, y)
    tr_col, _, roles = _treatment_columns(namer, rng, df, uc, tcol, ~np.isinf(starts), starts, times, staggered=True)
    query = CausalQuery("effect", treatment=tr_col, outcome=y)
    gold = {"sdid": 1.0, "did": 0.4}
    return _finish("panel_staggered", namer, rng, df, query, gold, float(eff[D].mean() * yc.scale),
                   {"n_units": n_units, "T": T, "cohorts": [int(s) for s in starts_pool], "time_kind": kind},
                   roles + (uc, tcol, y))


def time_series(rng) -> Scenario:
    """Один ряд без контрольной группы → прерванные временные ряды / CausalImpact."""
    namer = Namer(rng)
    kind = str(rng.choice(["date", "week", "month"]))
    # годовую сезонность можно отделить от тренда, только имея хотя бы два полных цикла
    T = int(rng.integers(*{"date": (40, 366), "week": (104, 261), "month": (36, 121)}[kind]))
    times, kind = time_axis(rng, kind, T)
    t0 = int(T * rng.uniform(0.6, 0.85))
    period = {"date": 7, "week": 52, "month": 12}[kind]
    rho, sigma = rng.uniform(0.2, 0.6), rng.uniform(0.1, 0.35)
    e = np.zeros(T)
    for i in range(1, T):
        e[i] = rho * e[i - 1] + sigma * rng.normal()
    base = np.linspace(0, rng.uniform(-1, 1), T) + rng.uniform(0, 0.5) * np.sin(2 * np.pi * np.arange(T) / period)
    tau = effect_size(rng, allow_zero=True)
    d = np.arange(T) >= t0
    yc = namer.sample(namer.theme.outcomes)[0]
    tcol = time_name(namer, kind)
    y = namer.pick(yc, ("Y", "y", "value"), "outcome")
    tc = namer.sample(namer.theme.treatments)[0]
    tr = namer.pick(tc, ("D", "intervention", "after"), "treatment")
    cols = {tcol: times, tr: _bin_no_labels(rng, d), y: positive(yc, yc.loc + yc.scale * (base + tau * d + e))}
    for c in namer.sample(namer.theme.covariates, int(rng.integers(0, 3))):
        if c.kind in ("cat", "bin"):
            continue
        cols[namer.covariate(c)] = positive(c, c.loc + c.scale * (0.7 * base + rng.normal(0, 0.5, T)))
    df = finalize(pd.DataFrame(cols), namer, rng, protected={tcol, tr, y}, cross_section=False)
    query = CausalQuery("effect", treatment=tr, outcome=y)
    gold = {"its": 1.0}
    return Scenario("time_series", df, query, gold, answer_mode(gold), float(tau * yc.scale), "ATT",
                    namer.theme.name, dict(namer.concepts), {"T": T, "t0": t0, "time_kind": kind, "rho": rho})
