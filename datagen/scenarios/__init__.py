"""Реестр сценариев и их доли в обучающем наборе."""
from __future__ import annotations

from typing import Callable

import numpy as np

from datagen.scenario import Scenario
from datagen.scenarios import cross_section as cs
from datagen.scenarios import discovery as disc
from datagen.scenarios import panel as pn

# имя -> (генератор, вес в смеси)
REGISTRY: dict[str, tuple[Callable[[np.random.Generator], Scenario], float]] = {
    "rct": (cs.rct, 1.0),
    "rct_heterogeneous": (cs.rct_heterogeneous, 0.6),
    "backdoor_small": (cs.backdoor_small, 1.0),
    "backdoor_large": (cs.backdoor_large, 1.0),
    "continuous_treatment": (cs.continuous_treatment, 0.8),
    "instrumental": (cs.instrumental, 0.8),
    "front_door": (cs.front_door, 0.5),
    "rdd": (cs.rdd, 0.8),
    "unidentifiable": (cs.unidentifiable, 0.6),
    "not_causal": (cs.not_causal, 0.5),
    "panel_parallel": (pn.panel_parallel, 1.0),
    "panel_factor": (pn.panel_factor, 1.0),
    "panel_staggered": (pn.panel_staggered, 0.7),
    "time_series": (pn.time_series, 0.8),
    "discovery_nongaussian": (disc.discovery_nongaussian, 0.5),
    "discovery_gaussian": (disc.discovery_gaussian, 0.4),
    "discovery_hidden": (disc.discovery_hidden, 0.3),
}


def generate(name: str, seed: int) -> Scenario:
    return REGISTRY[name][0](np.random.default_rng(seed))


def sample_name(rng: np.random.Generator) -> str:
    names = list(REGISTRY)
    w = np.array([REGISTRY[n][1] for n in names])
    return names[int(rng.choice(len(names), p=w / w.sum()))]
