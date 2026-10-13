"""Генерация вопросов пользователя по сценарию (EN — основной язык, RU — дополнительный).

Вопрос = вступление + основной вопрос + контекст, который нельзя увидеть в данных
(рандомизация, скрытый фактор, инструмент, медиатор, правило отбора) + завершение + шум.
Каждый шаблон основного вопроса имеет id; ~10% шаблонов отложены только для теста, чтобы мерить,
понимает ли модель формулировки, которых не видела.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from datagen.naming import THEMES

THEME = {t.name: t for t in THEMES}

# ---------- основные вопросы ----------

EFFECT_EN = [
    "Did {T} increase {Y}?", "Does {T} affect {Y}?", "What is the effect of {T} on {Y}?",
    "What's the causal effect of {T} on {Y}?", "How much did {T} change {Y}?",
    "Did {T} actually work? I want to know its impact on {Y}.", "Is there a causal impact of {T} on {Y}?",
    "Measure the impact of {T} on {Y}.", "Estimate how {T} influenced {Y}.",
    "I need to know whether {T} caused a change in {Y}.", "Does {T} lead to higher {Y}?",
    "Does {T} reduce {Y}?", "Can we say that {T} improves {Y}?", "Quantify the effect of {T} on {Y}.",
    "How big is the effect of {T} on {Y}?", "Would {Y} have been different without {T}?",
    "What happened to {Y} because of {T}?", "Is {T} worth it in terms of {Y}?",
    "Should we roll out {T}? Does it move {Y}?", "Help me figure out if {T} has any effect on {Y}.",
    "what's the impact of {T} on {Y}", "{T} -> {Y}: is there an effect?",
    "Did introducing {T} make a difference to {Y}?", "I'd like to evaluate {T}. Did it change {Y}?",
    "Which method should I use to estimate the effect of {T} on {Y}?",
    "How should I analyse whether {T} affects {Y}?",
    "What's the best way to measure the effect of {T} on {Y} with this data?",
    "Is the change in {Y} caused by {T} or is it just a coincidence?",
    "Does {T} have a positive effect on {Y}?", "Effect of {T} on {Y}?",
    "How do I find out if {T} drives {Y}?", "Did {Y} respond to {T}?",
]
EFFECT_PANEL_EN = [
    "Some {units} got {T} and others didn't. Did it change {Y}?",
    "{T} was introduced in some {units}. What did it do to {Y}?",
    "After {T}, did {Y} go up or down compared to what would have happened otherwise?",
    "We track {Y} for many {units} over time. How did {T} affect it?",
    "Compare {units} with and without {T}: what's the effect on {Y}?",
]
EFFECT_TS_EN = [
    "We launched {T} at some point. Did {Y} change because of it?",
    "Did {Y} change after {T} started?", "Was the jump in {Y} caused by {T}?",
    "We only have one time series. How do we measure the effect of {T} on {Y}?",
]
HETERO_EN = [
    "For whom does {T} work best in terms of {Y}?", "Does the effect of {T} on {Y} differ across {persons}?",
    "Which {persons} benefit most from {T}?", "Who should we target with {T} to maximise {Y}?",
    "Is the effect of {T} on {Y} the same for everyone?", "How does the impact of {T} on {Y} vary with {C}?",
    "I want individual-level effects of {T} on {Y}.", "Find segments where {T} increases {Y} the most.",
    "Which {persons} respond to {T}? We want to personalise it.",
    "Does {T} help some {persons} more than others when it comes to {Y}?",
    "Estimate heterogeneous effects of {T} on {Y}.",
]
DISCOVERY_EN = [
    "What are the causal relationships between these variables?", "Which variables cause which in this dataset?",
    "Can you find the causal graph behind this data?", "Build a causal DAG from this data.",
    "What drives what here?", "I don't have a hypothesis yet. What causes what among {VARS}?",
    "Discover the causal structure of {VARS}.", "Is {A} a cause of {B} or is it the other way around?",
    "Which of these variables are causes of {B}?", "How are {VARS} causally connected?",
    "Learn the causal structure from the data, please.",
]
NOT_CAUSAL_EN = [
    "Can you build a model to predict {Y}?", "Which features are most predictive of {Y}?",
    "Forecast {Y} for the next period.", "What's the correlation between {A} and {Y}?",
    "How accurately can we predict {Y} from the other columns?", "Train a model for {Y}.",
    "Which variables are most correlated with {Y}?", "Predict {Y} for new {persons}.",
    "Give me summary statistics of {Y} by {A}.", "What's the average {Y}?",
]

EFFECT_RU = [
    "Повлиял ли {T} на {Y}?", "Какой эффект {T} на {Y}?", "Как оценить влияние {T} на {Y}?",
    "Увеличил ли {T} {Y}?", "Есть ли причинный эффект {T} на {Y}?", "Насколько {T} изменил {Y}?",
    "Сработал ли {T}? Хочу понять, как он повлиял на {Y}.", "Каким методом оценить эффект {T} на {Y}?",
    "Что было бы с {Y} без {T}?", "Стоит ли раскатывать {T}? Влияет ли он на {Y}?",
    "Помоги понять, влияет ли {T} на {Y}.", "Снизил ли {T} {Y}?", "Как измерить эффект {T} на {Y} по этим данным?",
    "{T} -> {Y}: есть эффект?", "Изменение {Y} вызвано {T} или это совпадение?",
]
EFFECT_PANEL_RU = [
    "В части {units_ru} ввели {T}, в остальных нет. Изменился ли {Y}?",
    "Как {T} повлиял на {Y}, если сравнить с тем, что было бы без него?",
]
EFFECT_TS_RU = ["В какой-то момент мы запустили {T}. Изменился ли {Y} из-за этого?",
                "Изменился ли {Y} после запуска {T}?"]
HETERO_RU = [
    "Для кого {T} работает лучше всего (по {Y})?", "Отличается ли эффект {T} на {Y} для разных групп?",
    "Кому стоит давать {T}, чтобы {Y} вырос сильнее?", "Как эффект {T} на {Y} зависит от {C}?",
    "Хочу оценить индивидуальный эффект {T} на {Y}.",
]
DISCOVERY_RU = [
    "Какие причинно-следственные связи между этими переменными?", "Что на что влияет в этих данных?",
    "Построй причинный граф по этим данным.", "{A} влияет на {B} или наоборот?",
    "Какие из переменных являются причинами {B}?",
]
NOT_CAUSAL_RU = [
    "Можно ли построить модель для прогноза {Y}?", "Какие признаки лучше всего предсказывают {Y}?",
    "Спрогнозируй {Y} на следующий период.", "Какая корреляция между {A} и {Y}?",
    "Какие переменные сильнее всего связаны с {Y}?",
]

# ---------- контекст ----------

RANDOMIZED_EN = ["It was a randomized A/B test.", "{T} was assigned at random.", "We ran a randomized experiment.",
                 "Assignment to {T} was random.", "This data comes from an RCT."]
NOT_RANDOMIZED_EN = ["Note that {T} was not randomized: {persons} chose it themselves.",
                     "This is observational data, {T} wasn't assigned randomly.",
                     "{persons} opted into {T} on their own."]
HIDDEN_EN = ["But {H} probably affects both {T} and {Y}, and we didn't measure it.",
             "I worry that unmeasured {H} drives both {T} and {Y}.",
             "We don't observe {H}, which likely influences who gets {T} and also {Y}."]
INSTRUMENT_EN = ["{Z} shifts {T} but shouldn't affect {Y} directly.",
                 "We also have {Z}, which only affects {Y} through {T}.",
                 "{Z} influences {T} and is unrelated to everything else that affects {Y}."]
MEDIATOR_EN = ["{T} works only through {M}.", "The whole effect of {T} on {Y} goes through {M}.",
               "{T} changes {M}, and {M} is what changes {Y}; nothing else connects them."]
RUNNING_EN = ["{T} was given to everyone with {X} of at least {CUT}.", "Eligibility for {T} was {X} >= {CUT}.",
              "Only {persons} with {X} above {CUT} could get {T}."]
HIDDEN_DISC_EN = ["There may be unmeasured variables that affect several of these.",
                  "Not all relevant factors are in the data.", "Some common causes might be missing."]

RANDOMIZED_RU = ["Это был рандомизированный A/B-тест.", "{T} назначали случайно.", "Мы провели рандомизированный эксперимент."]
NOT_RANDOMIZED_RU = ["{T} не рандомизировали, люди выбирали сами.", "Это наблюдательные данные, {T} назначали не случайно."]
HIDDEN_RU = ["Но {H}, скорее всего, влияет и на {T}, и на {Y}, а мы это не измеряли.",
             "Боюсь, что неизмеренный фактор ({H}) влияет и на {T}, и на {Y}."]
INSTRUMENT_RU = ["{Z} влияет на {T}, но на {Y} напрямую не влияет.", "Есть ещё {Z}: он влияет на {Y} только через {T}."]
MEDIATOR_RU = ["{T} действует только через {M}.", "Весь эффект {T} на {Y} идёт через {M}."]
RUNNING_RU = ["{T} получали все, у кого {X} не ниже {CUT}.", "Право на {T} давалось при {X} >= {CUT}."]
HIDDEN_DISC_RU = ["Возможно, есть неизмеренные переменные, влияющие сразу на несколько из этих.",
                  "Не все важные факторы есть в данных."]

INTRO_EN = ["", "", "", "Hi!", "Hello,", "Quick question:", "My manager asked me this:", "Hey,", "Need help:",
            "I'm analysing our {theme} data.", "We have data on {persons}.", "I have a CSV with {n} rows."]
CLOSE_EN = ["", "", "", "Thanks!", "Which method should I use?", "What approach would you recommend?",
            "How should I analyse this?", "Any advice?", "What's the right methodology here?"]
INTRO_RU = ["", "", "Привет!", "Вопрос:", "Подскажи,", "Есть данные, нужен совет."]
CLOSE_RU = ["", "", "Спасибо!", "Какой метод выбрать?", "Как это правильно проанализировать?"]

HIDDEN_FACTORS = {
    "retail": [("local demand", "локальный спрос"), ("manager quality", "качество управляющего")],
    "marketing": [("customer motivation", "мотивация клиентов"), ("brand affinity", "лояльность к бренду")],
    "health": [("patients' health awareness", "забота пациентов о здоровье"), ("doctor preferences", "предпочтения врачей")],
    "education": [("student ability", "способности учеников"), ("parental involvement", "вовлечённость родителей")],
    "labor": [("worker ambition", "амбициозность сотрудников"), ("manager support", "поддержка руководителя")],
    "policy": [("local economic conditions", "местная экономическая ситуация"), ("political climate", "политический климат")],
    "tech": [("user engagement", "вовлечённость пользователей"), ("tech savviness", "техническая грамотность")],
}
UNITS_RU = {"retail": "магазинов", "marketing": "регионов", "health": "больниц", "education": "школ",
            "labor": "компаний", "policy": "штатов", "tech": "стран"}

TEMPLATE_SETS = {
    ("en", "effect"): EFFECT_EN, ("en", "effect_panel"): EFFECT_PANEL_EN, ("en", "effect_ts"): EFFECT_TS_EN,
    ("en", "heterogeneity"): HETERO_EN, ("en", "discovery"): DISCOVERY_EN, ("en", "not_causal"): NOT_CAUSAL_EN,
    ("ru", "effect"): EFFECT_RU, ("ru", "effect_panel"): EFFECT_PANEL_RU, ("ru", "effect_ts"): EFFECT_TS_RU,
    ("ru", "heterogeneity"): HETERO_RU, ("ru", "discovery"): DISCOVERY_RU, ("ru", "not_causal"): NOT_CAUSAL_RU,
}


def is_heldout(template_id: str) -> bool:
    """~10% шаблонов только для теста (детерминированно по id)."""
    return int(hashlib.md5(template_id.encode()).hexdigest(), 16) % 10 == 0


# ---------- упоминание колонок ----------

def _is_anon(rec: dict, col: str) -> bool:
    p = rec["concepts"].get(col, "")
    return p.startswith("variable ") or p in ("the treatment", "the outcome", "")


def mention(rec: dict, col: str, rng, lang: str, article: bool = True) -> str:
    """Как пользователь называет колонку: человеческой фразой, именем колонки или обоими."""
    phrase = rec["concepts"].get(col, col)
    code = f"`{col}`" if rng.random() < 0.7 else col
    if lang == "ru" or _is_anon(rec, col):
        return code
    r = rng.random()
    if r < 0.5:
        return ("the " if article and rng.random() < 0.6 else "") + phrase
    if r < 0.85:
        return code
    return f"{phrase} ({code})"


def plural(word: str) -> str:
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + ("es" if word.endswith(("s", "x", "ch", "sh")) else "s")


def persons(rec: dict) -> str:
    return plural(THEME[rec["theme"]].person.phrase)


def units(rec: dict) -> str:
    return plural(THEME[rec["theme"]].unit.phrase)


# ---------- шум ----------

def add_typos(text: str, rng, protected: set[str], n_typos: int) -> str:
    words = text.split(" ")
    idx = [i for i, w in enumerate(words)
           if len(w) > 3 and w.isalpha() and "`" not in w and w.strip(".,?!:").lower() not in protected]
    for i in rng.choice(idx, size=min(n_typos, len(idx)), replace=False) if idx else []:
        w = list(words[i])
        j = int(rng.integers(1, len(w) - 1))
        op = rng.choice(["swap", "drop", "dup"])
        if op == "swap":
            w[j], w[j + 1 if j + 1 < len(w) else j - 1] = w[j + 1 if j + 1 < len(w) else j - 1], w[j]
        elif op == "drop":
            del w[j]
        else:
            w.insert(j, w[j])
        words[i] = "".join(w)
    return " ".join(words)


def casualize(text: str, rng, protected: set[str]) -> str:
    if rng.random() < 0.2:
        text = add_typos(text, rng, protected, int(rng.integers(1, 3)))
    if rng.random() < 0.15:
        # нижний регистр, кроме имён колонок в обратных кавычках
        parts = re.split(r"(`[^`]*`)", text)
        text = "".join(p if p.startswith("`") else p.lower() for p in parts)
    if rng.random() < 0.15:
        text = text.rstrip("?.!")
    return re.sub(r"\s+", " ", text).strip()


# ---------- сборка вопроса ----------

def choose_lang(rec: dict, rng) -> str:
    cyrillic = any(re.search("[а-яё]", c, re.I) for c in rec["concepts"])
    return "ru" if rng.random() < (0.6 if cyrillic else 0.12) else "en"


def _pick_template(lang: str, kind: str, rng, allow_heldout: bool) -> tuple[str, str]:
    pool = TEMPLATE_SETS[(lang, kind)]
    ids = [f"{lang}:{kind}:{i}" for i in range(len(pool))]
    allowed = [i for i, tid in enumerate(ids) if allow_heldout or not is_heldout(tid)]
    i = int(rng.choice(allowed))
    return pool[i], ids[i]


def _fill(template: str, slots: dict) -> str:
    out = template
    for k, v in slots.items():
        out = out.replace("{" + k + "}", str(v))
    return out[0].upper() + out[1:] if out and out[0].islower() and not out.startswith("`") else out


def make_question(rec: dict, rng, lang: str | None = None, allow_heldout: bool = False) -> dict:
    """rec — запись сценария из datagen.build. Возвращает текст, язык и id шаблона."""
    lang = lang or choose_lang(rec, rng)
    q = rec["query"]
    qtype = q["question_type"]
    roles = {q.get(k) for k in ("treatment", "outcome", "instrument", "mediator", "running_variable")} - {None}
    others = [c for c in rec["concepts"] if c not in roles]
    slots = {"persons": persons(rec), "units": units(rec), "units_ru": UNITS_RU[rec["theme"]],
             "theme": rec["theme"], "n": rec["n_rows"]}

    if qtype in ("effect", "heterogeneity"):
        slots["T"] = mention(rec, q["treatment"], rng, lang)
        slots["Y"] = mention(rec, q["outcome"], rng, lang, article=False)
        kind = qtype
        if qtype == "effect" and rec["structure"] == "panel" and rng.random() < 0.4:
            kind = "effect_panel"
        elif qtype == "effect" and rec["structure"] == "time_series" and rng.random() < 0.5:
            kind = "effect_ts"
        if others:
            slots["C"] = mention(rec, str(rng.choice(others)), rng, lang, article=False)
        else:
            slots["C"] = "their characteristics" if lang == "en" else "характеристик"
    elif qtype == "discovery":
        nodes = rec["true_graph"]["nodes"]
        pick = list(rng.choice(nodes, size=min(len(nodes), int(rng.integers(2, 5))), replace=False))
        mentioned = [mention(rec, c, rng, lang, article=False) for c in pick]
        slots["VARS"] = (", ".join(mentioned[:-1]) + (" and " if lang == "en" else " и ") + mentioned[-1])
        slots["A"], slots["B"] = mentioned[0], mentioned[1]
        kind = "discovery"
    else:  # not_causal
        target = rec["params"]["target"]
        slots["Y"] = mention(rec, target, rng, lang, article=False)
        a = [c for c in others if c != target] or [target]
        slots["A"] = mention(rec, str(rng.choice(a)), rng, lang, article=False)
        kind = "not_causal"

    core, template_id = _pick_template(lang, kind, rng, allow_heldout)
    parts = [_fill(core, slots)]

    en = lang == "en"
    ctx = []
    if q.get("randomized") is True:
        ctx.append(rng.choice(RANDOMIZED_EN if en else RANDOMIZED_RU))
    elif q.get("randomized") is False and rng.random() < 0.7:
        ctx.append(rng.choice(NOT_RANDOMIZED_EN if en else NOT_RANDOMIZED_RU))
    if q.get("hidden_confounding") and qtype != "discovery":
        h_en, h_ru = HIDDEN_FACTORS[rec["theme"]][int(rng.integers(2))]
        slots["H"] = h_en if en else h_ru
        ctx.append(rng.choice(HIDDEN_EN if en else HIDDEN_RU))
    if q.get("hidden_confounding") and qtype == "discovery":
        ctx.append(rng.choice(HIDDEN_DISC_EN if en else HIDDEN_DISC_RU))
    if q.get("instrument"):
        slots["Z"] = mention(rec, q["instrument"], rng, lang)
        ctx.append(rng.choice(INSTRUMENT_EN if en else INSTRUMENT_RU))
    if q.get("mediator"):
        slots["M"] = mention(rec, q["mediator"], rng, lang, article=False)
        ctx.append(rng.choice(MEDIATOR_EN if en else MEDIATOR_RU))
    if q.get("running_variable"):
        slots["X"] = mention(rec, q["running_variable"], rng, lang, article=False)
        slots["CUT"] = f"{rec['params']['cutoff']:.3g}"
        ctx.append(rng.choice(RUNNING_EN if en else RUNNING_RU))
    rng.shuffle(ctx)
    parts += [_fill(c, slots) for c in ctx]

    intro = _fill(str(rng.choice(INTRO_EN if en else INTRO_RU)), slots) if True else ""
    close = str(rng.choice(CLOSE_EN if en else CLOSE_RU))
    text = " ".join(p for p in [intro, *parts, close] if p)
    protected = {c.lower() for c in rec["concepts"]}
    return {"text": casualize(text, rng, protected), "lang": lang, "template_id": template_id}
