"""Сквозной пайплайн: CSV + вопрос → профиль → разбор вопроса → диагностики → правила → методы → объяснение.

Бэкенд подключаемый: BaselineBackend (классический ML, этап 4) или NeuralBackend — наш трансформер
(этап 8, нужен PyTorch и чекпойнт). У бэкенда два метода: parse (проход 1) и rank (проход 2); нейросеть
ещё пишет объяснение (explain). Всё остальное — общий детерминированный код.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from advisor.confidence import CALIBRATION, Calibration, Decision
from advisor.diagnostics import DiagResult, run_diagnostics
from advisor.profiler import DataCard, profile
from advisor.query import CausalQuery
from advisor.rules import Verdict, evaluate

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class Backend(Protocol):
    name: str

    def parse(self, question: str, card_text: str) -> dict: ...

    def rank(self, structure: str, diagnostics: dict, verdicts: dict, fields: dict,
             question: str = "", card_text: str = "") -> tuple[dict, str]: ...


class BaselineBackend:
    name = "baseline"

    def __init__(self, path: Path = MODELS_DIR / "baseline.joblib"):
        bundle = joblib.load(path)
        self.parser, self.ranker = bundle["parser"], bundle["ranker"]

    def parse(self, question, card_text):
        return self.parser.predict(question, card_text)

    def rank(self, structure, diagnostics, verdicts, fields, question="", card_text=""):
        return self.ranker.predict(structure, diagnostics, verdicts, fields)


class NeuralBackend:
    """CausalGPT-mini: проход 1 — разбор с ограниченным декодированием (колонки только из этого CSV),
    проход 2 — вероятности методов из softmax, режим ответа по калиброванной уверенности
    (advisor/confidence.py, файл models/calibration.json), объяснение — генерация (model/decode.py)."""
    name = "causalgpt"

    def __init__(self, ckpt: Path = MODELS_DIR / "causalgpt.pt", device=None, calibration: Path | None = CALIBRATION):
        from model.decode import Advisor, ContextOverflow
        cal = Calibration.load(calibration, model=Path(ckpt)) if calibration is not None else Calibration()
        self.model = Advisor(ckpt, device, cal)
        self.overflow = ContextOverflow
        self.last: Decision | None = None  # решение последнего rank: уверенность, режим по токену модели

    @property
    def calibrated(self) -> bool:
        return bool(self.model.calibration.model_sha)

    def parse(self, question, card_text):
        try:
            return self.model.parse(question, card_text)
        except self.overflow as e:
            raise AdviceError(str(e)) from e

    def rank(self, structure, diagnostics, verdicts, fields, question="", card_text=""):
        self.last = None
        if fields.get("type") == "not_causal":
            return {}, "refuse"
        forbidden = {m for m, v in verdicts.items() if v["status"] == "forbidden"}
        try:
            self.last = self.model.decide(question, card_text, fields, _diag(diagnostics, verdicts, fields), forbidden)
        except self.overflow as e:
            raise AdviceError(str(e)) from e
        return self.last.probs, self.last.mode

    def explain(self, question, card_text, fields, diagnostics, verdicts, probs, mode) -> str | None:
        """Объяснение модели или None, если в нём есть число не из входа или не назван лучший метод."""
        from model.decode import explanation_ok

        diag = _diag(diagnostics, verdicts, fields)
        try:
            text = self.model.explain(question, card_text, fields, diag, probs, mode)
        except self.overflow:
            return None
        top = next(iter(probs)) if probs and mode != "refuse" else None
        return text if explanation_ok(text, "\n".join((card_text, question, diag)), top) else None


def _diag(diagnostics: dict, verdicts: dict, fields: dict) -> str:
    from datagen.explanations import diag_text
    return diag_text({"diagnostics": diagnostics, "verdicts": verdicts,
                      "query": {"running_variable": fields.get("running")}})


def load_backend(name: str = "baseline", **kw) -> Backend:
    return {"baseline": BaselineBackend, "causalgpt": NeuralBackend}[name](**kw)


class AdviceError(ValueError):
    """Понятная пользователю ошибка (например, не удалось найти воздействие в вопросе)."""


@dataclass
class Advice:
    lang: str
    fields: dict
    card: DataCard
    diagnostics: dict[str, DiagResult]
    verdicts: list[Verdict]
    methods: dict[str, float]  # метод -> вероятность (после маскировки правилами)
    mode: str  # recommend | options | refuse
    explanation: str
    backend: str
    notes: list[str] = field(default_factory=list)
    explanation_source: str = "template"  # model — написала нейросеть, template — шаблон из фактов
    confidence: float | None = None  # калиброванная вероятность, что лучший метод верен (только нейросеть)


def detect_lang(question: str) -> str:
    """Русский, если среди букв вне `имён колонок` заметная доля кириллицы."""
    text = re.sub(r"`[^`]*`", " ", question)
    cyr = len(re.findall(r"[а-яё]", text, re.I))
    lat = len(re.findall(r"[a-z]", text, re.I))
    return "ru" if cyr > 0.4 * max(cyr + lat, 1) else "en"


def query_from_fields(fields: dict, columns: list[str]) -> CausalQuery:
    qtype = fields.get("type", "effect")
    for k in ("treatment", "outcome", "instrument", "mediator", "running", "target"):
        if fields.get(k) is not None and fields[k] not in columns:
            raise AdviceError(f"column '{fields[k]}' is not in the data")
    if qtype == "not_causal":
        return CausalQuery("not_causal")
    if qtype == "discovery":
        return CausalQuery("discovery", hidden_confounding=True if fields.get("hidden") == "yes" else None)
    if not fields.get("treatment") or not fields.get("outcome"):
        raise AdviceError("could not identify the treatment and the outcome; please choose them explicitly")
    if fields["treatment"] == fields["outcome"]:
        raise AdviceError("treatment and outcome must be different columns")
    rnd = fields.get("randomized")
    return CausalQuery(qtype, treatment=fields["treatment"], outcome=fields["outcome"],
                       instrument=fields.get("instrument"), mediator=fields.get("mediator"),
                       running_variable=fields.get("running"),
                       randomized=None if rnd is None else rnd == "yes",
                       hidden_confounding=True if fields.get("hidden") == "yes" else None)


def diag_dict(diags: dict[str, DiagResult]) -> dict:
    return {k: {"applicable": r.applicable, "values": r.values, "flags": r.flags, "message": r.message}
            for k, r in diags.items()}


def verdict_dict(verdicts: list[Verdict]) -> dict:
    return {v.method: {"status": v.status, "reasons": v.reasons, "pros": v.pros, "cons": v.cons} for v in verdicts}


def weights_for_explanation(probs: dict[str, float], mode: str) -> dict[str, float]:
    """Вероятности → веса в шкале эталона (лучший = 1.0), согласованные с режимом ответа."""
    if not probs:
        return {}
    top = max(probs.values())
    w = {m: p / top for m, p in probs.items()}
    ranked = sorted(w, key=w.get, reverse=True)
    if mode == "options" and len(ranked) > 1:
        w[ranked[1]] = 1.0
    elif mode == "recommend" and len(ranked) > 1 and w[ranked[1]] >= 0.9:
        w[ranked[1]] = 0.85
    return w


def advise(df: pd.DataFrame, question: str, backend: Backend, fields: dict | None = None) -> Advice:
    from datagen.explanations import explain

    card = profile(df)
    card_text = card.to_text()
    notes = []
    if fields is None:
        fields = backend.parse(question, card_text)
    query = query_from_fields(fields, list(df.columns))
    diags = run_diagnostics(df, card, query)
    verdicts = evaluate(card, query, diags)
    dd, vd = diag_dict(diags), verdict_dict(verdicts)
    probs, mode = backend.rank(card.structure, dd, vd, fields, question=question, card_text=card_text)
    if query.question_type != "not_causal" and not probs:
        mode = "refuse"
        notes.append("no method is applicable to this data and question")
    decision = getattr(backend, "last", None)
    confidence = decision.confidence if decision is not None and getattr(backend, "calibrated", False) else None
    lang = detect_lang(question)
    rec = {"query": {"question_type": query.question_type, "treatment": query.treatment,
                     "outcome": query.outcome, "hidden_confounding": query.hidden_confounding},
           "gold": weights_for_explanation(probs, mode), "answer_mode": mode, "diagnostics": dd,
           "verdicts": vd, "params": {"target": fields.get("target")}}
    text, source = None, "template"
    # отказы модель училась объяснять только скрытым конфаундером: другой отказ объясняет шаблон
    model_can_explain = probs and (mode != "refuse" or query.hidden_confounding)
    if hasattr(backend, "explain") and query.question_type != "not_causal" and model_can_explain:
        text = backend.explain(question, card_text, fields, dd, vd, probs, mode)
        if text is not None:
            source = "model"
        else:
            notes.append("the model's own explanation failed the fact check, so a template one is shown")
    if text is None:
        seed = int(hashlib.md5(question.encode()).hexdigest(), 16) % (2 ** 32)
        text = explain(rec, lang, np.random.default_rng(seed))
    return Advice(lang, fields, card, diags, verdicts, probs, mode, text, backend.name, notes, source, confidence)
