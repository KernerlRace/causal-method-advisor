"""Бейзлайн без нейросети (этап 4): классический ML поверх тех же входов, что увидит модель.

- QuestionParser — проход 1: тип вопроса, контекст (рандомизация, скрытый фактор) и роли колонок.
  Тип и контекст: TF-IDF (символьные n-граммы устойчивы к опечаткам) + логистическая регрессия.
  Роли: бустинг по парам «вопрос × колонка» на признаках упоминания, сходства и слов-подсказок.
- MethodRanker — проход 2: вероятности методов и режим ответа по признакам диагностик, флагам,
  вердиктам правил и разбору вопроса. Правила потом маскируют запрещённые методы.

Это нижняя планка: нейросеть должна её превзойти, а до тех пор бейзлайн обслуживает демо.
"""
from __future__ import annotations

import math
import re

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from advisor.knowledge import FLAGS, load_cards
from advisor.profiler import name_tokens

QTYPES = ("effect", "heterogeneity", "discovery", "not_causal")
ROLES = ("none", "treatment", "outcome", "instrument", "mediator", "running")
KINDS = ("binary", "integer", "numeric", "categorical", "datetime", "text", "constant")
CARD_COL = re.compile(r"^col (.+) (" + "|".join(KINDS) + r") uniq=(\d+)")

CUES = {
    "instrument": ("shift", "directly", "only affects", "unrelated", "lottery", "напрямую", "только через"),
    "mediator": ("through", "goes through", "works only", "via", "через"),
    "running": ("at least", ">=", "above", "eligib", "не ниже", "право"),
    "outcome": (" on ", " на ", "in terms of", "predict", "forecast", "прогноз", "предсказ"),
    "treatment": ("effect of", "impact of", "did ", "does ", "introduc", "launch", "эффект", "повлиял", "влияние"),
}


def card_columns(card_text: str) -> list[tuple[str, str, int]]:
    """Колонки из текста профиля: (имя, тип, число уникальных)."""
    out = []
    for line in card_text.splitlines():
        if m := CARD_COL.match(line):
            out.append((m.group(1), m.group(2), int(m.group(3))))
    return out


def _norm(text: str) -> str:
    return re.sub(r"[`_\-]+", " ", text.lower())


def _stem(w: str) -> str:
    return w[:5] if len(w) > 5 else w


# ---------- проход 1 ----------

class QuestionParser:
    def __init__(self):
        self.tfidf = FeatureUnion([
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True,
                                      max_features=60000)),
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000)),
        ])
        self.heads = {k: LogisticRegression(C=4.0, max_iter=3000) for k in ("type", "randomized", "hidden")}
        self.roles = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1)

    # признаки одной пары «вопрос × колонка»
    @staticmethod
    def column_features(question: str, col: str, kind: str, n_unique: int, qtype_p: np.ndarray,
                        all_cols: list[str]) -> list[float]:
        q, qn = question.lower(), _norm(question)
        toks = [t for t in name_tokens(col) if len(t) >= 2]
        qwords = re.findall(r"\w+", qn)
        qstems = {_stem(w) for w in qwords}
        exact = float(col.lower() in q)
        tick = float(f"`{col.lower()}`" in q)
        phrase = float(" ".join(toks) in qn) if toks else 0.0
        frac = (sum(_stem(t) in qstems for t in toks) / len(toks)) if toks else 0.0
        # позиция первого упоминания (по полному имени или первому совпавшему слову)
        pos = q.find(col.lower())
        if pos < 0:
            for t in toks:
                m = re.search(r"\b" + re.escape(_stem(t)), qn)
                if m:
                    pos = m.start()
                    break
        rel_pos = pos / max(len(q), 1) if pos >= 0 else -1.0
        window = qn[max(0, pos - 45): pos + len(col) + 45] if pos >= 0 else ""
        before = qn[max(0, pos - 25): pos] if pos >= 0 else ""
        cues = [float(any(c in window for c in CUES[r])) for r in ("instrument", "mediator", "running")]
        cues += [float(any(c in before for c in CUES[r])) for r in ("outcome", "treatment")]
        # порядок упоминания среди всех упомянутых колонок
        positions = sorted(p for c in all_cols if (p := q.find(c.lower())) >= 0)
        order = positions.index(pos) if pos in positions else -1
        kind_1h = [float(kind == k) for k in KINDS]
        return [exact, tick, phrase, frac, rel_pos, float(order), *cues, *kind_1h,
                math.log1p(n_unique), *qtype_p]

    def _texts(self, examples):
        return [e["question"] for e in examples]

    def fit(self, examples: list[dict]):
        X = self.tfidf.fit_transform(self._texts(examples))
        y_type = [e["parse_fields"]["type"] for e in examples]
        y_rand = [e["parse_fields"].get("randomized", "unknown") for e in examples]
        y_hid = [e["parse_fields"].get("hidden", "no") for e in examples]
        for head, y in zip(("type", "randomized", "hidden"), (y_type, y_rand, y_hid)):
            self.heads[head].fit(X, y)
        P = self.heads["type"].predict_proba(X)
        rows, labels = [], []
        for e, p in zip(examples, P):
            cols = card_columns(e["card"])
            names = [c for c, _, _ in cols]
            role_of = {v: ("outcome" if k == "target" else k) for k, v in e["parse_fields"].items()
                       if k in ("treatment", "outcome", "instrument", "mediator", "running", "target")}
            for c, kind, nu in cols:
                rows.append(self.column_features(e["question"], c, kind, nu, self._qtype_vec(p), names))
                labels.append(role_of.get(c, "none"))
        self.roles.fit(np.array(rows), np.array(labels))
        return self

    def _qtype_vec(self, p: np.ndarray) -> np.ndarray:
        classes = list(self.heads["type"].classes_)
        return np.array([p[classes.index(t)] if t in classes else 0.0 for t in QTYPES])

    def predict(self, question: str, card_text: str) -> dict:
        X = self.tfidf.transform([question])
        p_type = self.heads["type"].predict_proba(X)[0]
        qtype = self.heads["type"].classes_[int(np.argmax(p_type))]
        fields: dict = {"type": str(qtype)}
        rand = self.heads["randomized"].predict(X)[0]
        if rand != "unknown":
            fields["randomized"] = str(rand)
        if self.heads["hidden"].predict(X)[0] == "yes":
            fields["hidden"] = "yes"
        cols = card_columns(card_text)
        if not cols or qtype == "discovery":
            return fields
        names = [c for c, _, _ in cols]
        F = np.array([self.column_features(question, c, k, nu, self._qtype_vec(p_type), names) for c, k, nu in cols])
        P = self.roles.predict_proba(F)
        cls = list(self.roles.classes_)
        used: set[str] = set()

        def take(role: str, required: bool) -> str | None:
            if role not in cls:
                return None
            order = np.argsort(-P[:, cls.index(role)])
            for i in order:
                if names[i] in used:
                    continue
                if required or P[i, cls.index(role)] > 0.5:
                    used.add(names[i])
                    return names[i]
                return None
            return None

        if qtype == "not_causal":
            if (t := take("outcome", True)):
                fields["target"] = t
            return fields
        for role, required in (("treatment", True), ("outcome", True), ("instrument", False),
                               ("mediator", False), ("running", False)):
            if (c := take(role, required)):
                fields[role] = c
        return fields


# ---------- проход 2 ----------

NUMERIC = [  # (диагностика, значение, преобразование)
    ("sample_size", "n_rows", "log"), ("sample_size", "n_covariates", "id"),
    ("sample_size", "n_treated_units", "log"), ("sample_size", "n_control_units", "log"),
    ("sample_size", "n_pre_periods", "log"), ("sample_size", "n_post_periods", "log"),
    ("pretrend_test", "drift_relative_to_sd", "id"), ("pretrend_test", "p_value", "nlog"),
    ("sc_pre_fit", "pre_r2", "clip"), ("sc_pre_fit", "outside_hull_share", "id"),
    ("overlap", "propensity_auc", "id"), ("overlap", "extreme_propensity_share", "id"),
    ("overlap", "max_abs_smd", "id"), ("running_variable", "assignment_accuracy", "id"),
    ("running_variable", "jump_at_cutoff", "id"), ("first_stage_strength", "first_stage_F", "log"),
    ("normality", "non_gaussian_share", "id"), ("nonlinearity", "r2_linear", "clip"),
    ("nonlinearity", "r2_boosting", "clip"), ("nonlinearity", "gain", "id"),
]
STATUS = {"allowed": 0.0, "warning": 1.0, "forbidden": 2.0}


def _tr(x, how):
    if x is None:
        return np.nan
    x = float(x)
    return {"log": math.log1p(max(x, 0)), "nlog": -math.log10(max(x, 1e-300)),
            "clip": max(min(x, 1.0), -1.0), "id": x}[how]


def ranker_features(structure: str, diagnostics: dict, verdicts: dict, fields: dict) -> np.ndarray:
    """diagnostics/verdicts — в словарной форме, как в записях сценариев."""
    methods = sorted(load_cards())
    flags = {f for r in diagnostics.values() for f in r.get("flags", [])}
    if fields.get("hidden") == "yes":
        flags.add("hidden_confounding")
    x = [float(structure == s) for s in ("cross_section", "panel", "time_series")]
    x += [float(fields.get("type") == t) for t in QTYPES]
    x += [float(k in fields) for k in ("instrument", "mediator", "running")]
    x += [float(fields.get("randomized", "unknown") == v) for v in ("yes", "no", "unknown")]
    x += [float(f in flags) for f in sorted(FLAGS)]
    for diag, key, how in NUMERIC:
        r = diagnostics.get(diag)
        x.append(_tr(r["values"].get(key), how) if r and r.get("applicable") else np.nan)
    x += [STATUS[verdicts[m]["status"]] if m in verdicts else 2.0 for m in methods]
    return np.array(x, dtype=float)


class MethodRanker:
    """По модели на каждый метод: предсказывает его вес в эталоне (1.0 = лучший выбор, 0 = не подходит).
    Так ранжируются все методы, а не только лучший: второй хороший вариант тоже получает свой вес."""

    def __init__(self):
        self.methods = sorted(load_cards())
        self.models: dict = {}
        self.mode_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1)

    def fit(self, records: list[dict]):
        from datagen.explanations import parse_fields
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, Y, modes = [], [], []
        for r in records:
            x = ranker_features(r["structure"], r["diagnostics"], r["verdicts"], parse_fields(r))
            X.append(x)
            Y.append([r["gold"].get(m, 0.0) for m in self.methods])
            modes.append(r["answer_mode"])
        X, Y = np.array(X), np.array(Y)
        for j, m in enumerate(self.methods):
            if Y[:, j].max() > 0:
                self.models[m] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1).fit(X, Y[:, j])
        self.mode_clf.fit(X, np.array(modes))
        return self

    def predict(self, structure: str, diagnostics: dict, verdicts: dict, fields: dict) -> tuple[dict, str]:
        """Возвращает (метод -> доля веса, по убыванию; сумма = 1) и режим ответа."""
        if fields.get("type") == "not_causal":
            return {}, "refuse"
        x = ranker_features(structure, diagnostics, verdicts, fields)[None, :]
        w = {m: float(np.clip(model.predict(x)[0], 0.0, 1.0)) for m, model in self.models.items()}
        # правила маскируют запрещённое; совсем слабые кандидаты не показываем
        w = {m: v for m, v in w.items() if verdicts.get(m, {}).get("status") != "forbidden" and v > 0.05}
        total = sum(w.values()) or 1.0
        probs = {str(m): v / total for m, v in sorted(w.items(), key=lambda kv: -kv[1])}
        return probs, str(self.mode_clf.predict(x)[0])
