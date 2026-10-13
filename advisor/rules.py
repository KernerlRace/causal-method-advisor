"""Rule engine: жёсткие ограничения из карточек методов.

Правила не ранжируют методы, а только отвечают на вопрос «можно ли вообще» и собирают
аргументы за и против (good_flags / bad_flags). Ранжирование — задача модели (и бейзлайна этапа 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from advisor.diagnostics import DiagResult, all_flags
from advisor.knowledge import MethodCard, load_cards
from advisor.profiler import DataCard
from advisor.query import CausalQuery

ALLOWED, WARNING, FORBIDDEN = "allowed", "warning", "forbidden"
_STATUS_ORDER = {ALLOWED: 0, WARNING: 1, FORBIDDEN: 2}

# понятные названия порогов для сообщений
_THRESHOLD_NAMES = {
    "min_rows": "rows", "min_treated_units": "treated units", "min_control_units": "control units",
    "min_pre_periods": "pre-treatment periods", "min_post_periods": "post-treatment periods",
    "min_numeric_cols": "numeric variables",
}


@dataclass
class Facts:
    """Всё, что правила знают о задаче: из профиля, запроса и диагностик."""
    question_type: str
    structure: str
    treatment_type: str | None
    counts: dict  # имя порога -> фактическое значение (None = неприменимо)
    available: set  # instrument / mediator / running_variable
    randomized: bool | None
    flags: set


@dataclass
class Verdict:
    method: str
    status: str
    reasons: list[str] = field(default_factory=list)  # почему запрещён / какие оговорки
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


def treatment_type(card: DataCard, query: CausalQuery) -> str | None:
    if not query.treatment:
        return None
    if card.structure == "panel" and card.design_for(query.treatment) is not None:
        return "binary"  # в панели воздействие — индикатор D_it
    col = card.column(query.treatment)
    if col.kind == "binary":
        return "binary"
    if col.kind == "categorical" or (col.kind == "integer" and col.n_unique <= 10):
        return "categorical"
    return "continuous"


def build_facts(card: DataCard, query: CausalQuery, diags: dict[str, DiagResult]) -> Facts:
    design = card.design_for(query.treatment) if card.structure == "panel" else None
    counts = {
        "min_rows": card.n_rows,
        "min_numeric_cols": len(card.names("numeric", "integer", "binary")),
        "min_treated_units": design.n_treated_units if design else None,
        "min_control_units": design.n_control_units if design else None,
        "min_pre_periods": design.n_pre_periods if design else None,
        "min_post_periods": design.n_post_periods if design else None,
    }
    if card.structure == "time_series":
        sv = diags["sample_size"].values if "sample_size" in diags else {}
        counts["min_pre_periods"] = sv.get("n_pre_periods", card.n_rows)
        counts["min_post_periods"] = sv.get("n_post_periods")
    available = {k for k in ("instrument", "mediator", "running_variable") if getattr(query, k)}
    flags = all_flags(diags)
    if "running_variable_found" in flags:
        available.add("running_variable")
    if query.hidden_confounding:
        flags.add("hidden_confounding")
    randomized = query.randomized
    return Facts(query.question_type, card.structure, treatment_type(card, query), counts,
                 available, randomized, flags)


def evaluate_method(card: MethodCard, facts: Facts) -> Verdict:
    v = Verdict(card.id, ALLOWED)
    hard, soft = v.reasons, []
    req = card.requires

    if facts.question_type not in card.question_types:
        hard.append(f"answers {'/'.join(card.question_types)} questions, not '{facts.question_type}'")
    if facts.structure not in req["structure"]:
        hard.append(f"needs {' or '.join(req['structure'])} data, but the data is {facts.structure}")
    if req.get("treatment_types") and facts.treatment_type and facts.treatment_type not in req["treatment_types"]:
        hard.append(f"needs a {' or '.join(req['treatment_types'])} treatment, but it is {facts.treatment_type}")
    for need in req.get("needs", []):
        if need not in facts.available:
            name = need.replace("_", " ")
            hard.append(f"needs {'an' if name[0] in 'aeiou' else 'a'} {name}, none is specified or detected")
    if req.get("randomized"):
        if facts.randomized is False:
            hard.append("needs randomized treatment assignment, but the treatment was not randomized")
        elif facts.randomized is None:
            soft.append("valid only if treatment was randomized; please confirm")

    for key, minimum in req.items():
        if key in _THRESHOLD_NAMES and (actual := facts.counts.get(key)) is not None and actual < minimum:
            hard.append(f"needs at least {minimum} {_THRESHOLD_NAMES[key]}, has {actual}")
    for key, minimum in card.recommended.items():
        if (actual := facts.counts.get(key)) is not None and actual < minimum:
            soft.append(f"works best with at least {minimum} {_THRESHOLD_NAMES[key]}, has {actual}")

    v.pros = [text for flag, text in card.good_flags.items() if flag in facts.flags]
    v.cons = [text for flag, text in card.bad_flags.items() if flag in facts.flags]

    if hard:
        v.status = FORBIDDEN
    elif soft or v.cons:
        v.status = WARNING
        v.reasons.extend(soft)
    return v


def evaluate(card: DataCard, query: CausalQuery, diags: dict[str, DiagResult],
             cards: dict[str, MethodCard] | None = None) -> list[Verdict]:
    """Вердикт по каждому методу. Порядок: допустимые, с оговорками, запрещённые;
    внутри группы — по балансу аргументов (это не рекомендация, а удобный порядок для чтения)."""
    cards = cards or load_cards()
    if query.question_type == "not_causal":
        return [Verdict(m, FORBIDDEN, ["the question is not causal (prediction or description)"])
                for m in cards]
    facts = build_facts(card, query, diags)
    verdicts = [evaluate_method(c, facts) for c in cards.values()]
    return sorted(verdicts, key=lambda v: (_STATUS_ORDER[v.status], len(v.cons) - len(v.pros), v.method))
