"""Сценарии без времени: эксперименты, back-door, непрерывное воздействие, IV, front-door, RDD,
неидентифицируемый эффект и «не причинный» вопрос."""
from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.query import CausalQuery
from datagen.naming import Namer, realize
from datagen.scenario import (Scenario, answer_mode, backdoor_gold, covariates, effect_size, encode_binary, finalize,
                              loguniform_int, outcome_factor, overlap_gold, positive, prune_gold, scale_outcome)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _roles(namer: Namer, continuous: bool = False):
    th = namer.theme
    t_concept = namer.sample(th.cont_treatments if continuous else th.treatments)[0]
    y_concept = namer.sample(th.outcomes)[0]
    t = namer.pick(t_concept, ("T", "treatment", "treat", "d"), "treatment")
    y = namer.pick(y_concept, ("Y", "y", "outcome", "target"), "outcome")
    return t, t_concept, y, y_concept


def _build(name, namer, rng, cols, query, gold, true_effect, estimand, params, refuse=False,
           protected=(), extras=None) -> Scenario:
    df = finalize(pd.DataFrame(cols), namer, rng, protected=set(protected) | query.role_columns(),
                  cross_section=True)
    gold = prune_gold(gold, len(df))
    return Scenario(name, df, query, gold, answer_mode(gold, refuse), true_effect, estimand,
                    namer.theme.name, dict(namer.concepts), params, extras=extras or {})


# ---------- эксперименты ----------

def rct(rng, heterogeneous: bool = False) -> Scenario:
    namer = Namer(rng)
    n = loguniform_int(rng, 1000 if heterogeneous else 120, 30000 if heterogeneous else 20000)
    k = int(rng.integers(2, 8))
    cols, F = covariates(namer, rng, n, k)
    t, _, y, yc = _roles(namer)
    multi_arm = not heterogeneous and rng.random() < 0.15
    p_treat = rng.uniform(0.2, 0.5)
    beta = rng.normal(0, 0.5, k)
    tau = effect_size(rng, allow_zero=not heterogeneous)
    noise = rng.uniform(0.5, 1.5)
    if multi_arm:
        arms = rng.choice([["control", "variant_a", "variant_b"], ["A", "B", "C"], ["0", "1", "2"]])
        a = rng.choice(3, size=n, p=[1 - p_treat, p_treat / 2, p_treat / 2])
        effects = np.array([0.0, tau, tau * rng.uniform(0.3, 1.5)])
        y_std = F @ beta + effects[a] + noise * rng.normal(size=n)
        cols[t] = np.array(arms)[a]
        true_effect = float(effects[1] * outcome_factor(yc, y_std))
        params = {"arms": list(arms)}
    else:
        d = rng.random(n) < p_treat
        if heterogeneous:
            j = int(rng.integers(k))
            gamma = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
            tau_i = tau + gamma * F[:, j] + (0.3 * np.maximum(F[:, (j + 1) % k], 0) if rng.random() < 0.5 else 0)
        else:
            tau_i = np.full(n, tau)
        y_std = F @ beta + tau_i * d + noise * rng.normal(size=n)
        cols[t] = encode_binary(rng, d)
        true_effect = float(tau_i.mean() * outcome_factor(yc, y_std))
        params = {"p_treat": float(p_treat)}
    cols[y] = scale_outcome(yc, y_std)
    randomized = True if rng.random() < 0.7 else None
    if heterogeneous:
        query = CausalQuery("heterogeneity", treatment=t, outcome=y, randomized=randomized)
        gold = {"meta_learners": 1.0, "causal_forest": 1.0, "dml": 0.6, "doubly_robust": 0.6}
        extras = {"true_cate": tau_i * outcome_factor(yc, y_std)}
    else:
        query = CausalQuery("effect", treatment=t, outcome=y, randomized=randomized)
        gold = {"diff_in_means": 1.0, "regression": 0.7}
        extras = {}
    params.update(n=n, k=k, tau_std=tau, multi_arm=multi_arm, noise=float(noise))
    return _build("rct_heterogeneous" if heterogeneous else "rct", namer, rng, cols, query, gold,
                  true_effect, "ATE", params, extras=extras)


def rct_heterogeneous(rng) -> Scenario:
    return rct(rng, heterogeneous=True)


# ---------- back-door ----------

def backdoor_small(rng) -> Scenario:
    """Мало данных, линейные связи, все конфаундеры наблюдаемы → простая регрессия."""
    namer = Namer(rng)
    n = loguniform_int(rng, 80, 600)
    k = int(rng.integers(2, 7))
    cols, F = covariates(namer, rng, n, k)
    t, _, y, yc = _roles(namer)
    a = rng.uniform(0.3, 0.8, k) * rng.choice([-1, 1], k)
    d = rng.random(n) < _sigmoid(F @ a)
    tau = effect_size(rng)
    beta = rng.uniform(0.3, 1.0, k) * np.sign(a) * rng.choice([1, 1, -1], k)
    y_std = F @ beta + tau * d + rng.uniform(0.5, 1.0) * rng.normal(size=n)
    cols[t], cols[y] = encode_binary(rng, d), scale_outcome(yc, y_std)
    query = CausalQuery("effect", treatment=t, outcome=y, randomized=False if rng.random() < 0.3 else None)
    return _build("backdoor_small", namer, rng, cols, query, backdoor_gold(n, k), tau * outcome_factor(yc, y_std),
                  "ATE", {"n": n, "k": k, "tau_std": tau})


def backdoor_large(rng) -> Scenario:
    """Много данных, нелинейный конфаундинг → DML / дважды робастная оценка."""
    namer = Namer(rng)
    n = loguniform_int(rng, 3000, 40000)
    k = int(rng.integers(5, 26))
    cols, F = covariates(namer, rng, n, k)
    t, _, y, yc = _roles(namer)
    m = min(k, 5)  # первые m ковариат — нелинейные конфаундеры
    g = np.column_stack([np.sin(2 * F[:, 0]), F[:, 1] ** 2 - 1, np.abs(F[:, 2]) - 0.8,
                         F[:, 3] * F[:, 4] if m >= 5 else F[:, 3], np.maximum(F[:, 0], 0)])[:, :m]
    a = rng.uniform(0.5, 1.0, g.shape[1]) * rng.choice([-1, 1], g.shape[1])
    ps = np.clip(_sigmoid(g @ a + 0.3 * F[:, :m].sum(axis=1) * rng.choice([-1, 1])), 0.03, 0.97)
    d = rng.random(n) < ps
    tau = effect_size(rng)
    b = 1.2 * np.sign(a) * rng.uniform(0.6, 1.2, g.shape[1])
    y_std = g @ b + 0.3 * F @ rng.normal(0, 0.5, F.shape[1]) + tau * d + rng.uniform(0.5, 1.0) * rng.normal(size=n)
    cols[t], cols[y] = encode_binary(rng, d), scale_outcome(yc, y_std)
    query = CausalQuery("effect", treatment=t, outcome=y, randomized=False if rng.random() < 0.3 else None)
    return _build("backdoor_large", namer, rng, cols, query, backdoor_gold(n, k), tau * outcome_factor(yc, y_std),
                  "ATE", {"n": n, "k": k, "tau_std": tau})


def backdoor_targeted(rng) -> Scenario:
    """Воздействие назначали по нескольким признакам (адресное предложение, лечение по тяжести
    состояния): все конфаундеры наблюдаемы, но у многих объектов воздействие почти предрешено —
    группы плохо пересекаются. Эталон тогда — варианты (scenario.overlap_gold, по флагу диагностики);
    при умеренной силе отбора флага нет, и ответ как у back-door сценария того же размера.

    Правило отбора — сумма 2–4 ковариат с близкими весами: ни одна переменная в одиночку его не
    объясняет, поэтому это не порог RDD. Наклон логита от 1.5 (около 5% объектов с propensity
    за пределами [0.05, 0.95]) до 8 (около 70%)."""
    namer = Namer(rng)
    n = loguniform_int(rng, 200, 20000)
    k = int(rng.integers(2, 9))
    cols, F = covariates(namer, rng, n, k)
    t, _, y, yc = _roles(namer)
    m = int(rng.integers(2, min(k, 4) + 1))  # сколько ковариат входит в правило отбора
    a = rng.uniform(0.7, 1.0, m) * rng.choice([-1, 1], m)
    score = F[:, :m] @ a
    score = (score - score.mean()) / max(score.std(), 1e-9)
    slope = float(np.exp(rng.uniform(np.log(1.5), np.log(8.0))))
    d = rng.random(n) < _sigmoid(slope * (score - rng.uniform(-0.6, 0.6)))
    tau = effect_size(rng)
    beta = rng.uniform(0.3, 1.0, k) * rng.choice([1, 1, -1], k)
    beta[:m] = np.abs(beta[:m]) * np.sign(a)  # отбирали тех, у кого исход и так выше (или ниже)
    nonlinear = rng.random() < 0.3
    y_std = (F @ beta + (0.6 * np.sin(2 * F[:, 0]) + 0.4 * (F[:, 1] ** 2 - 1) if nonlinear else 0)
             + tau * d + rng.uniform(0.5, 1.0) * rng.normal(size=n))
    cols[t], cols[y] = encode_binary(rng, d), scale_outcome(yc, y_std)
    query = CausalQuery("effect", treatment=t, outcome=y, randomized=False if rng.random() < 0.3 else None)
    return _build("backdoor_targeted", namer, rng, cols, query, overlap_gold(n, k, nonlinear),
                  tau * outcome_factor(yc, y_std), "ATE",
                  {"n": n, "k": k, "tau_std": tau, "m": m, "slope": slope, "nonlinear": nonlinear})


def continuous_treatment(rng) -> Scenario:
    namer = Namer(rng)
    linear = rng.random() < 0.4
    n = loguniform_int(rng, 300 if linear else 1500, 20000)
    k = int(rng.integers(2, 12))
    cols, F = covariates(namer, rng, n, k)
    t, tc, y, yc = _roles(namer, continuous=True)
    a = rng.uniform(0.3, 0.8, k) * rng.choice([-1, 1], k)
    conf = F @ a if linear else F @ a + np.sin(2 * F[:, 0]) + (F[:, 1] ** 2 - 1 if k > 1 else 0)
    t_std = 0.8 * conf / max(conf.std(), 1e-9) + rng.normal(size=n)
    theta = effect_size(rng)
    y_std = theta * t_std + (F @ (np.sign(a) * 0.7) if linear else 1.2 * conf) + rng.uniform(0.5, 1) * rng.normal(size=n)
    cols[t] = positive(tc, tc.loc + tc.scale * t_std)
    cols[y] = scale_outcome(yc, y_std)
    query = CausalQuery("effect", treatment=t, outcome=y)
    gold = ({"regression": 1.0, "dml": 0.8} if linear
            else {"dml": 1.0, "causal_forest": 0.6, "regression": 0.3})
    return _build("continuous_treatment", namer, rng, cols, query, gold,
                  theta * outcome_factor(yc, y_std) / tc.scale, "ATE per unit of treatment",
                  {"n": n, "k": k, "linear": linear, "theta_std": theta})


# ---------- скрытый конфаундинг ----------

def instrumental(rng) -> Scenario:
    namer = Namer(rng)
    n = loguniform_int(rng, 800, 30000)
    k = int(rng.integers(1, 5))
    cols, F = covariates(namer, rng, n, k)
    u = rng.normal(size=n)  # ненаблюдаемый конфаундер
    binary_t = rng.random() < 0.4
    t, tc, y, yc = _roles(namer, continuous=not binary_t)
    zc = namer.sample(namer.theme.instruments)[0]
    z_std = rng.normal(size=n)
    z_vals = realize(zc, z_std, rng)
    z_eff = (z_vals - z_vals.mean()) / z_vals.std()
    z = namer.pick(zc, ("Z", "z", "instrument", "iv"), "instrument")
    strength = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
    lin = strength * z_eff + u + 0.3 * F.sum(axis=1)
    beta = effect_size(rng)
    if binary_t:
        d = lin + rng.normal(size=n) > 0
        y_std = beta * d + 1.5 * u + 0.4 * F.sum(axis=1) + 0.5 * rng.normal(size=n)
        cols[t] = encode_binary(rng, d)
        true_effect = beta * outcome_factor(yc, y_std)
    else:
        t_std = lin + rng.normal(size=n)
        y_std = beta * t_std + 1.5 * u + 0.4 * F.sum(axis=1) + 0.5 * rng.normal(size=n)
        cols[t] = positive(tc, tc.loc + tc.scale * t_std)
        true_effect = beta * outcome_factor(yc, y_std) / tc.scale
    cols[z], cols[y] = z_vals, scale_outcome(yc, y_std)
    query = CausalQuery("effect", treatment=t, outcome=y, instrument=z, hidden_confounding=True)
    gold = {"iv": 1.0, "sensitivity_analysis": 0.4}
    return _build("instrumental", namer, rng, cols, query, gold, true_effect, "LATE",
                  {"n": n, "binary_treatment": binary_t, "strength": strength})


def front_door(rng) -> Scenario:
    namer = Namer(rng)
    n = loguniform_int(rng, 1000, 20000)
    cols, F = covariates(namer, rng, n, int(rng.integers(0, 3)))
    u = rng.normal(size=n)
    t, _, y, yc = _roles(namer)
    mc = namer.sample(namer.theme.mediators)[0]
    m = namer.pick(mc, ("M", "m", "mediator"), "mediator")
    d = (u + rng.normal(size=n)) > 0
    a = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    m_std = a * d + rng.normal(size=n)
    b = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
    y_std = b * m_std + 1.5 * u + 0.5 * rng.normal(size=n)
    cols[t], cols[m] = encode_binary(rng, d), positive(mc, mc.loc + mc.scale * m_std)
    cols[y] = scale_outcome(yc, y_std)
    query = CausalQuery("effect", treatment=t, outcome=y, mediator=m, hidden_confounding=True)
    gold = {"front_door": 1.0, "sensitivity_analysis": 0.4}
    return _build("front_door", namer, rng, cols, query, gold, a * b * outcome_factor(yc, y_std), "ATE", {"n": n, "a": a, "b": b})


def unidentifiable(rng) -> Scenario:
    """Сильный скрытый конфаундер, нет ни инструмента, ни медиатора → честный отказ."""
    namer = Namer(rng)
    n = loguniform_int(rng, 200, 20000)
    k = int(rng.integers(1, 6))
    cols, F = covariates(namer, rng, n, k)
    u = rng.normal(size=n)
    t, _, y, yc = _roles(namer)
    d = rng.random(n) < _sigmoid(1.2 * u + 0.4 * F.sum(axis=1))
    tau = effect_size(rng)
    y_std = tau * d + 1.5 * u + 0.4 * F.sum(axis=1) + 0.5 * rng.normal(size=n)
    cols[t], cols[y] = encode_binary(rng, d), scale_outcome(yc, y_std)
    query = CausalQuery("effect", treatment=t, outcome=y, hidden_confounding=True, randomized=False)
    gold = {"sensitivity_analysis": 1.0}
    return _build("unidentifiable", namer, rng, cols, query, gold, tau * outcome_factor(yc, y_std), "ATE", {"n": n, "k": k},
                  refuse=True)


# ---------- разрыв ----------

def rdd(rng) -> Scenario:
    namer = Namer(rng)
    n = loguniform_int(rng, 800, 20000)
    cols, F = covariates(namer, rng, n, int(rng.integers(0, 4)))
    rc = namer.sample(namer.theme.running)[0]
    x_std = rng.uniform(-2, 2, n) if rng.random() < 0.5 else rng.normal(size=n)
    cut_std = float(np.quantile(x_std, rng.uniform(0.3, 0.7)))
    fuzzy = rng.random() < 0.3
    above = x_std >= cut_std
    d = rng.random(n) < np.where(above, 0.9, 0.1) if fuzzy else above
    t, _, y, yc = _roles(namer)
    x = namer.pick(rc, ("X", "x", "running", "score"), "running_variable")
    shape = rng.choice(["linear", "quadratic", "cubic"])
    f = {"linear": 0.5 * x_std, "quadratic": 0.5 * x_std + 0.2 * x_std ** 2,
         "cubic": 0.4 * x_std + 0.1 * x_std ** 3}[shape]
    tau = effect_size(rng, allow_zero=False)
    y_std = f + tau * d + 0.2 * F.sum(axis=1) + rng.uniform(0.3, 0.7) * rng.normal(size=n)
    x_raw = rc.loc + rc.scale * x_std
    x_vals = positive(rc, x_raw)
    cutoff = rc.loc + rc.scale * cut_std + (x_vals[0] - x_raw[0])  # порог сдвигается вместе с переменной
    cols[x], cols[t], cols[y] = x_vals, encode_binary(rng, d), scale_outcome(yc, y_std)
    # в нечётком RDD правило отбора называет пользователь; в чётком — иногда тоже
    query = CausalQuery("effect", treatment=t, outcome=y,
                        running_variable=x if (fuzzy or rng.random() < 0.5) else None)
    gold = {"rdd": 1.0}
    return _build("rdd", namer, rng, cols, query, gold, tau * outcome_factor(yc, y_std), "LATE at the cutoff",
                  {"n": n, "fuzzy": fuzzy, "cutoff": cutoff, "shape": str(shape)},
                  extras={"running": x, "cutoff": cutoff, "fuzzy": fuzzy})


# ---------- не про причинность ----------

def not_causal(rng) -> Scenario:
    base = (rct if rng.random() < 0.5 else backdoor_small)(rng)
    q = CausalQuery("not_causal")
    return Scenario("not_causal", base.df, q, {}, "refuse", None, None, base.theme, base.concepts,
                    {"target": base.query.outcome, "base": base.name})
