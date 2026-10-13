"""Сценарии поиска причинной структуры: случайный DAG, линейная SEM.

Варианты: негауссовский шум (LiNGAM ориентирует все рёбра), гауссовский (PC/GES дают класс
эквивалентности), скрытый общий предок (FCI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from advisor.query import CausalQuery
from datagen.naming import Namer
from datagen.scenario import Scenario, answer_mode, finalize, loguniform_int, positive


def random_dag(rng, p: int) -> np.ndarray:
    """W[i, j] != 0 означает ребро i -> j; порядок узлов — топологический."""
    W = np.zeros((p, p))
    prob = rng.uniform(0.3, 0.5)
    for j in range(1, p):
        parents = [i for i in range(j) if rng.random() < prob]
        if not parents:  # без изолированных узлов
            parents = [int(rng.integers(j))]
        for i in parents:
            W[i, j] = rng.uniform(0.5, 1.2) * rng.choice([-1, 1])
    return W


def _noise(rng, kind: str, n: int) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(size=n)
    choice = rng.choice(["uniform", "laplace", "exponential"])
    e = {"uniform": rng.uniform(-1, 1, n), "laplace": rng.laplace(size=n),
         "exponential": rng.exponential(size=n)}[choice]
    return (e - e.mean()) / e.std()


def discovery(rng, variant: str | None = None) -> Scenario:
    variant = variant or str(rng.choice(["nongaussian", "nongaussian", "gaussian", "hidden"]))
    namer = Namer(rng)
    p = int(rng.integers(4, 9))
    n = loguniform_int(rng, 500, 10000)
    p_total = p + (1 if variant == "hidden" else 0)
    W = random_dag(rng, p_total)
    hidden = None
    if variant == "hidden":
        # латентный корень с двумя детьми, сам в данные не попадает
        W[:, 0] = 0
        W[0, :] = 0
        kids = rng.choice(np.arange(1, p_total), size=2, replace=False)
        W[0, kids] = rng.uniform(0.8, 1.2, 2) * rng.choice([-1, 1], 2)
        hidden = 0
    noise_kind = "gaussian" if variant == "gaussian" else "nongaussian"
    X = np.zeros((n, p_total))
    for j in range(p_total):
        X[:, j] = X @ W[:, j] + _noise(rng, noise_kind, n)
    observed = [j for j in range(p_total) if j != hidden]
    pool = tuple(c for c in namer.theme.covariates + namer.theme.outcomes if c.kind == "num")
    concepts = namer.sample(pool, p)
    names = [namer.covariate(c) for c in concepts]
    cols = {}
    for name, c, j in zip(names, concepts, observed):
        z = (X[:, j] - X[:, j].mean()) / X[:, j].std()
        cols[name] = positive(c, c.loc + c.scale * z)
    perm = rng.permutation(p)  # порядок колонок не выдаёт топологический порядок
    df = pd.DataFrame({names[i]: cols[names[i]] for i in perm})
    df = finalize(df, namer, rng, protected=set(names), cross_section=True)
    df = df[[c for c in df.columns if c in names] + [c for c in df.columns if c not in names]]
    edges = [[names[observed.index(i)], names[observed.index(j)]]
             for i in observed for j in observed if W[i, j] != 0]
    confounded = ([[names[observed.index(int(k))] for k in np.flatnonzero(W[hidden])]]
                  if hidden is not None else [])
    gold = {"nongaussian": {"lingam": 1.0, "pc": 0.6, "ges": 0.5, "notears": 0.5, "fci": 0.5},
            "gaussian": {"pc": 1.0, "ges": 1.0, "notears": 0.6, "fci": 0.6, "lingam": 0.1},
            "hidden": {"fci": 1.0, "pc": 0.4, "ges": 0.3, "lingam": 0.2}}[variant]
    query = CausalQuery("discovery", hidden_confounding=True if variant == "hidden" else None)
    return Scenario("discovery_" + variant, df, query, gold, answer_mode(gold), None, "causal graph",
                    namer.theme.name, dict(namer.concepts),
                    {"n": n, "p": p, "variant": variant, "noise": noise_kind},
                    true_graph={"nodes": names, "edges": edges, "hidden_confounded_pairs": confounded})


def discovery_nongaussian(rng) -> Scenario:
    return discovery(rng, "nongaussian")


def discovery_gaussian(rng) -> Scenario:
    return discovery(rng, "gaussian")


def discovery_hidden(rng) -> Scenario:
    return discovery(rng, "hidden")
