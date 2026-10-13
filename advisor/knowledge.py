"""Загрузка и валидация карточек методов из knowledge/methods/*.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

METHODS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "methods"

QUESTION_TYPES = {"effect", "heterogeneity", "discovery"}
STRUCTURES = {"cross_section", "panel", "time_series"}
TREATMENT_TYPES = {"binary", "categorical", "continuous"}
NEEDS = {"instrument", "mediator", "running_variable"}
FAMILIES = {"rct", "cate", "backdoor", "iv", "frontdoor", "rdd", "panel", "timeseries",
            "discovery", "sensitivity"}
THRESHOLDS = {"min_rows", "min_treated_units", "min_control_units", "min_pre_periods",
              "min_post_periods", "min_numeric_cols"}
DIAGNOSTICS = {"balance", "overlap", "sample_size", "first_stage_strength", "running_variable",
               "pretrend_test", "sc_pre_fit", "normality"}
# все флаги, которые могут выставить диагностики или запрос; карточки ссылаются только на них
FLAGS = {
    "randomization_plausible", "imbalance", "good_overlap", "poor_overlap",
    "small_sample", "large_sample", "high_dim",
    "pretrend_ok", "pretrend_violated", "sc_good_fit", "sc_poor_fit", "treated_outside_hull",
    "single_treated_unit", "many_treated_units", "staggered", "short_pre_period",
    "few_controls", "no_controls", "has_controls", "policy_design",
    "running_variable_found", "sharp_cutoff", "weak_instrument",
    "gaussian_data", "non_gaussian", "hidden_confounding",
}

REQUIRED_KEYS = {"id", "name", "name_ru", "family", "question_types", "estimand", "summary",
                 "requires", "assumptions", "diagnostics", "libraries", "sources"}
OPTIONAL_KEYS = {"recommended", "good_flags", "bad_flags", "prefer_over"}


@dataclass(frozen=True)
class MethodCard:
    id: str
    name: str
    name_ru: str
    family: str
    question_types: tuple[str, ...]
    estimand: str
    summary: str
    requires: dict
    recommended: dict
    assumptions: tuple[str, ...]
    diagnostics: tuple[str, ...]
    good_flags: dict = field(default_factory=dict)
    bad_flags: dict = field(default_factory=dict)
    prefer_over: dict = field(default_factory=dict)
    libraries: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


class CardError(ValueError):
    pass


def _check_subset(card_id: str, what: str, values, allowed: set) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise CardError(f"{card_id}: unknown {what}: {sorted(unknown)}")


def validate(raw: dict, source: str) -> MethodCard:
    keys = set(raw)
    missing = REQUIRED_KEYS - keys
    if missing:
        raise CardError(f"{source}: missing keys {sorted(missing)}")
    _check_subset(source, "keys", keys, REQUIRED_KEYS | OPTIONAL_KEYS)

    cid = raw["id"]
    if Path(source).stem != cid:
        raise CardError(f"{source}: id '{cid}' does not match file name")
    if raw["family"] not in FAMILIES:
        raise CardError(f"{cid}: unknown family '{raw['family']}'")
    _check_subset(cid, "question_types", raw["question_types"], QUESTION_TYPES)
    _check_subset(cid, "diagnostics", raw["diagnostics"], DIAGNOSTICS)

    req = raw["requires"]
    _check_subset(cid, "requires keys", req,
                  THRESHOLDS | {"structure", "treatment_types", "needs", "randomized"})
    _check_subset(cid, "structure", req.get("structure", []), STRUCTURES)
    _check_subset(cid, "treatment_types", req.get("treatment_types", []), TREATMENT_TYPES)
    _check_subset(cid, "needs", req.get("needs", []), NEEDS)
    if not req.get("structure"):
        raise CardError(f"{cid}: requires.structure must not be empty")
    if "discovery" not in raw["question_types"] and not req.get("treatment_types"):
        raise CardError(f"{cid}: effect methods must declare requires.treatment_types")

    rec = raw.get("recommended") or {}
    _check_subset(cid, "recommended keys", rec, THRESHOLDS)
    for k, v in rec.items():
        if k in req and v < req[k]:
            raise CardError(f"{cid}: recommended.{k}={v} is below requires.{k}={req[k]}")

    for key in ("good_flags", "bad_flags"):
        _check_subset(cid, key, raw.get(key) or {}, FLAGS)

    return MethodCard(
        id=cid, name=raw["name"], name_ru=raw["name_ru"], family=raw["family"],
        question_types=tuple(raw["question_types"]), estimand=str(raw["estimand"]),
        summary=raw["summary"], requires=dict(req), recommended=dict(rec),
        assumptions=tuple(raw["assumptions"]), diagnostics=tuple(raw["diagnostics"]),
        good_flags=dict(raw.get("good_flags") or {}), bad_flags=dict(raw.get("bad_flags") or {}),
        prefer_over=dict(raw.get("prefer_over") or {}),
        libraries=tuple(raw["libraries"]), sources=tuple(raw["sources"]),
    )


@lru_cache(maxsize=1)
def load_cards(directory: Path = METHODS_DIR) -> dict[str, MethodCard]:
    cards: dict[str, MethodCard] = {}
    for path in sorted(directory.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            card = validate(yaml.safe_load(f), str(path))
        cards[card.id] = card
    # prefer_over должен ссылаться на существующие методы
    for card in cards.values():
        _check_subset(card.id, "prefer_over targets", card.prefer_over, set(cards))
    return cards
