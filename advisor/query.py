"""Структурированный причинный запрос. На этапе 1 заполняется вручную, позже его выдаёт модель."""
from __future__ import annotations

from dataclasses import dataclass

QUESTION_TYPES = ("effect", "heterogeneity", "discovery", "not_causal")


@dataclass
class CausalQuery:
    question_type: str  # effect | heterogeneity | discovery | not_causal
    treatment: str | None = None
    outcome: str | None = None
    covariates: list[str] | None = None  # None = все остальные подходящие колонки
    instrument: str | None = None
    mediator: str | None = None
    running_variable: str | None = None
    randomized: bool | None = None  # None = неизвестно
    hidden_confounding: bool | None = None  # пользователь подозревает скрытые конфаундеры

    def __post_init__(self):
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(f"unknown question_type '{self.question_type}'")
        if self.question_type in ("effect", "heterogeneity") and not (self.treatment and self.outcome):
            raise ValueError("effect questions need both treatment and outcome")

    def role_columns(self) -> set[str]:
        return {c for c in (self.treatment, self.outcome, self.instrument, self.mediator,
                            self.running_variable) if c}
