"""Целевой ответ для обучения: разбор вопроса (проход 1), текст диагностик (вход прохода 2),
методы с весами, режим ответа и объяснение (выход прохода 2).

Объяснение собирается из карточек методов, вердиктов правил и реальных чисел диагностик,
поэтому модель учится опираться на факты, а не выдумывать их.
"""
from __future__ import annotations

import numpy as np

from advisor.knowledge import load_cards

CARDS = load_cards()


def fmt(x: float) -> str:
    if x is None:
        return "?"
    if abs(x) >= 1000 or (abs(x) < 0.01 and x != 0):
        return f"{x:.3g}"
    return f"{x:.3g}" if abs(x) < 10 else f"{x:.1f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ---------- проход 1: разбор вопроса ----------

def parse_fields(rec: dict) -> dict:
    q = rec["query"]
    f = {"type": q["question_type"]}
    if q["question_type"] == "not_causal":
        f["target"] = rec["params"]["target"]
    for k, short in (("treatment", "treatment"), ("outcome", "outcome"), ("instrument", "instrument"),
                     ("mediator", "mediator"), ("running_variable", "running")):
        if q.get(k):
            f[short] = q[k]
    if q.get("randomized") is not None:
        f["randomized"] = "yes" if q["randomized"] else "no"
    if q.get("hidden_confounding"):
        f["hidden"] = "yes"
    return f


def parse_text(fields: dict) -> str:
    parts = [f"<T:{fields['type']}>"]
    # значения в обратных кавычках: имена колонок бывают с пробелами
    parts += [f"{k}=`{v}`" for k, v in fields.items() if k != "type"]
    return " ".join(parts)


# ---------- вход прохода 2: результаты инструментов ----------

def diag_text(rec: dict) -> str:
    d = rec["diagnostics"]
    lines = []
    s = d["sample_size"]["values"]
    line = f"size rows={s['n_rows']} covariates={s['n_covariates']}"
    if "n_treated_units" in s:
        line += (f" treated={s['n_treated_units']} controls={s['n_control_units']}"
                 f" pre={s['n_pre_periods']} post={s['n_post_periods']}")
    elif "n_pre_periods" in s:
        line += f" pre={s['n_pre_periods']} post={s['n_post_periods']}"
    lines.append(line)
    if (r := d.get("pretrend_test")) and r["applicable"]:
        v = r["values"]
        lines.append(f"pretrend drift={fmt(v['drift_over_pre_period'])} rel={fmt(v['drift_relative_to_sd'])} "
                     f"p={fmt(v['p_value'])}")
    if (r := d.get("sc_pre_fit")) and r["applicable"]:
        v = r["values"]
        lines.append(f"scfit r2={fmt(v['pre_r2'])} donors={v['n_donors_used']} outside={fmt(v['outside_hull_share'])}")
    if (r := d.get("overlap")) and r["applicable"]:
        v = r["values"]
        lines.append(f"overlap auc={fmt(v['propensity_auc'])} extreme={fmt(v['extreme_propensity_share'])} "
                     f"maxsmd={fmt(v['max_abs_smd'])}")
    if (r := d.get("running_variable")) and r["applicable"] and (
            "running_variable_found" in r["flags"] or rec["query"].get("running_variable")):
        v = r["values"]
        lines.append(f"threshold var={v['variable']} cutoff={fmt(v['cutoff'])} "
                     f"acc={fmt(v['assignment_accuracy'])} jump={fmt(v['jump_at_cutoff'])}")
    if (r := d.get("first_stage_strength")) and r["applicable"]:
        lines.append(f"firststage F={fmt(r['values']['first_stage_F'])}")
    if (r := d.get("nonlinearity")) and r["applicable"]:
        v = r["values"]
        lines.append(f"nonlinearity r2_linear={fmt(v['r2_linear'])} r2_boosting={fmt(v['r2_boosting'])}")
    if (r := d.get("normality")) and r["applicable"]:
        lines.append(f"normality nongaussian={fmt(r['values']['non_gaussian_share'])}")
    flags = sorted({f for r in d.values() for f in r["flags"]})
    lines.append("flags " + (" ".join(flags) or "-"))
    v = rec["verdicts"]
    lines.append("allowed " + (" ".join(m for m, x in v.items() if x["status"] == "allowed") or "-"))
    lines.append("warning " + (" ".join(m for m, x in v.items() if x["status"] == "warning") or "-"))
    return "\n".join(lines)


# ---------- факты для объяснения ----------

FLAG_RU = {
    "randomization_plausible": "ковариаты сбалансированы, рандомизация выглядит правдоподобно",
    "imbalance": "группы заметно различаются по ковариатам",
    "good_overlap": "группы хорошо пересекаются по ковариатам",
    "poor_overlap": "у части объектов вероятность воздействия почти 0 или почти 1",
    "small_sample": "данных мало", "large_sample": "данных много", "high_dim": "ковариат много",
    "pretrend_ok": "тренды до воздействия параллельны", "pretrend_violated": "тренды до воздействия не параллельны",
    "sc_good_fit": "синтетический контроль хорошо повторяет траекторию до воздействия",
    "sc_poor_fit": "синтетический контроль плохо повторяет траекторию до воздействия",
    "treated_outside_hull": "объект с воздействием лежит вне диапазона контрольных",
    "single_treated_unit": "воздействие получил всего один объект",
    "many_treated_units": "объектов с воздействием много", "staggered": "воздействие вводили в разное время",
    "short_pre_period": "мало периодов до воздействия", "few_controls": "мало контрольных объектов",
    "no_controls": "контрольной группы нет", "has_controls": "есть контрольные объекты",
    "policy_design": "данные устроены как «группа × период»",
    "running_variable_found": "воздействие назначено по порогу непрерывной переменной",
    "sharp_cutoff": "порог определяет воздействие почти точно", "weak_instrument": "инструмент слабый",
    "gaussian_data": "данные близки к нормальным", "non_gaussian": "данные заметно негауссовские",
    "hidden_confounding": "есть подозрение на скрытый конфаундер",
    "nonlinear": "исход зависит от ковариат нелинейно", "linear_ok": "связи близки к линейным",
}


def facts(rec: dict, lang: str) -> list[str]:
    """Предложения с цифрами диагностик (в порядке важности)."""
    d, en, out = rec["diagnostics"], lang == "en", []
    if (r := d.get("pretrend_test")) and r["applicable"]:
        v = r["values"]
        if "pretrend_violated" in r["flags"]:
            out.append(f"Before the treatment the gap between treated and control units drifted by "
                       f"{fmt(v['drift_over_pre_period'])} ({pct(v['drift_relative_to_sd'])} of the outcome's SD, "
                       f"p={fmt(v['p_value'])}), so the trends are not parallel." if en else
                       f"До воздействия разрыв между группами сместился на {fmt(v['drift_over_pre_period'])} "
                       f"({pct(v['drift_relative_to_sd'])} от SD исхода, p={fmt(v['p_value'])}), тренды не параллельны.")
        else:
            out.append(f"Pre-treatment trends look parallel (drift {fmt(v['drift_over_pre_period'])}, "
                       f"p={fmt(v['p_value'])})." if en else
                       f"Тренды до воздействия выглядят параллельными (смещение {fmt(v['drift_over_pre_period'])}, "
                       f"p={fmt(v['p_value'])}).")
    if (r := d.get("sc_pre_fit")) and r["applicable"]:
        v = r["values"]
        good = v["pre_r2"] >= 0.75
        s = ((f"A synthetic control reproduces the treated path before the treatment well (R²={fmt(v['pre_r2'])})"
              if good else f"A synthetic control fits the treated path before the treatment poorly "
                           f"(R²={fmt(v['pre_r2'])})") if en else
             (f"Синтетический контроль хорошо повторяет траекторию до воздействия (R²={fmt(v['pre_r2'])})"
              if good else f"Синтетический контроль плохо повторяет траекторию до воздействия (R²={fmt(v['pre_r2'])})"))
        if v["outside_hull_share"] > 0.5:
            s += (f", but the treated unit is outside the range of the controls in "
                  f"{pct(v['outside_hull_share'])} of pre-periods" if en else
                  f", но объект с воздействием вне диапазона контрольных в {pct(v['outside_hull_share'])} периодов")
        out.append(s + ".")
    if (r := d.get("running_variable")) and r["applicable"] and "running_variable_found" in r["flags"]:
        v = r["values"]
        out.append(f"Treatment switches at `{v['variable']}` >= {fmt(v['cutoff'])}: this rule explains "
                   f"{pct(v['assignment_accuracy'])} of assignments." if en else
                   f"Воздействие включается при `{v['variable']}` >= {fmt(v['cutoff'])}: это правило объясняет "
                   f"{pct(v['assignment_accuracy'])} назначений.")
    if (r := d.get("first_stage_strength")) and r["applicable"]:
        F = r["values"]["first_stage_F"]
        strong = F >= 10
        out.append((f"The instrument is {'strong' if strong else 'weak'}: first-stage F={fmt(F)}." if en else
                    f"Инструмент {'сильный' if strong else 'слабый'}: F первой стадии = {fmt(F)}."))
    if (r := d.get("nonlinearity")) and r["applicable"]:
        v = r["values"]
        if "nonlinear" in r["flags"]:
            out.append(f"The outcome depends on the covariates non-linearly: boosting reaches R²={fmt(v['r2_boosting'])} "
                       f"vs {fmt(v['r2_linear'])} for a linear model." if en else
                       f"Исход зависит от ковариат нелинейно: бустинг даёт R²={fmt(v['r2_boosting'])}, "
                       f"линейная модель {fmt(v['r2_linear'])}.")
        else:
            out.append(f"A linear model fits the outcome about as well as boosting "
                       f"(R²={fmt(v['r2_linear'])} vs {fmt(v['r2_boosting'])})." if en else
                       f"Линейная модель описывает исход почти так же хорошо, как бустинг "
                       f"(R²={fmt(v['r2_linear'])} против {fmt(v['r2_boosting'])}).")
    if (r := d.get("overlap")) and r["applicable"]:
        v = r["values"]
        out.append(f"Covariates predict the treatment with AUC={fmt(v['propensity_auc'])}; "
                   f"{pct(v['extreme_propensity_share'])} of units have extreme propensity scores." if en else
                   f"Ковариаты предсказывают воздействие с AUC={fmt(v['propensity_auc'])}; "
                   f"у {pct(v['extreme_propensity_share'])} объектов экстремальный propensity score.")
    if (r := d.get("normality")) and r["applicable"]:
        share = r["values"]["non_gaussian_share"]
        out.append(f"{pct(share)} of the continuous variables are clearly non-Gaussian." if en else
                   f"{pct(share)} непрерывных переменных заметно негауссовские.")
    s = d["sample_size"]["values"]
    if "n_treated_units" in s:
        nt = s["n_treated_units"]
        out.append(f"There {'is' if nt == 1 else 'are'} {nt} treated unit{'' if nt == 1 else 's'} and "
                   f"{s['n_control_units']} control units, "
                   f"{s['n_pre_periods']} periods before and {s['n_post_periods']} after the treatment." if en else
                   f"Объектов с воздействием {s['n_treated_units']}, контрольных {s['n_control_units']}; "
                   f"периодов до воздействия {s['n_pre_periods']}, после {s['n_post_periods']}.")
    elif "n_pre_periods" in s:
        out.append(f"This is a single series with no control group: {s['n_pre_periods']} periods before the "
                   f"intervention and {s['n_post_periods']} after." if en else
                   f"Это один ряд без контрольной группы: {s['n_pre_periods']} периодов до вмешательства "
                   f"и {s['n_post_periods']} после.")
    else:
        out.append(f"The data has {s['n_rows']} rows." if en else f"В данных {s['n_rows']} строк.")
    return out


def _flags(rec: dict) -> set:
    return {f for r in rec["diagnostics"].values() for f in r["flags"]} | (
        {"hidden_confounding"} if rec["query"].get("hidden_confounding") else set())


def name(mid: str, lang: str) -> str:
    return CARDS[mid].name if lang == "en" else CARDS[mid].name_ru


def pros(rec: dict, mid: str, lang: str) -> list[str]:
    if lang == "en":
        return list(rec["verdicts"][mid]["pros"])
    return [FLAG_RU[f] for f in CARDS[mid].good_flags if f in _flags(rec)]


def cons(rec: dict, mid: str, lang: str) -> list[str]:
    if lang == "en":
        return list(rec["verdicts"][mid]["cons"])
    return [FLAG_RU[f] for f in CARDS[mid].bad_flags if f in _flags(rec)]


def assumptions(mid: str, lang: str, rng, k: int = 2) -> list[str]:
    pool = list(CARDS[mid].assumptions if lang == "en" else CARDS[mid].assumptions_ru)
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def _join(items: list[str], lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + (" and " if lang == "en" else " и ") + items[-1]


def _caveats(rec: dict, mid: str, lang: str) -> list[str]:
    reasons = rec["verdicts"][mid]["reasons"] if rec["verdicts"][mid]["status"] != "forbidden" else []
    if lang == "en":
        return [r[0].upper() + r[1:] + "." for r in reasons]
    out = []
    for r in reasons:
        if "randomized" in r:
            out.append("Метод корректен, только если воздействие назначали случайно, это стоит подтвердить.")
    return out


def _avoid(rec: dict, best: set, lang: str, rng) -> str | None:
    """Метод, который напрашивается, но здесь хуже: низкий вес в gold и есть аргументы против."""
    cands = [m for m, w in rec["gold"].items() if w < 0.5 and m not in best and cons(rec, m, lang)]
    if not cands:
        cands = [m for m, v in rec["verdicts"].items()
                 if v["status"] == "warning" and m not in best and m not in rec["gold"] and cons(rec, m, lang)]
    if not cands:
        return None
    m = str(rng.choice(cands))
    c = cons(rec, m, lang)[0]
    return (f"{rng.choice(['Not recommended', 'Avoid', 'Be careful with'])}: {name(m, lang)}, because {c}."
            if lang == "en" else f"{name(m, lang)} здесь хуже: {c}.")


# ---------- объяснение ----------

def _refuse_no_method(rec: dict, lang: str) -> str:
    """Отказ не из-за скрытого конфаундера: правила запретили всё, что подошло бы по виду данных и вопросу
    (в обучающих данных таких отказов нет — там отказ всегда про скрытый конфаундер)."""
    en = lang == "en"
    t, y = rec["query"]["treatment"], rec["query"]["outcome"]
    near = {m: v["reasons"] for m, v in rec["verdicts"].items()
            if v["status"] == "forbidden" and v["reasons"]
            and not any(r.startswith(("answers ", "needs cross_section", "needs panel", "needs time_series"))
                        for r in v["reasons"])}
    fact = facts(rec, lang)[-1]
    if en:
        why = " ".join(f"{name(m, lang)} {v[0]}." for m, v in list(near.items())[:3])
        return (f"No method can estimate the effect of `{t}` on `{y}` from this data. {fact} {why} "
                f"What you can do: collect data on comparable units that did not get the treatment, or a longer "
                f"history before it, and ask again.").replace("  ", " ")
    names = _join([name(m, lang) for m in list(near)[:3]], lang)
    return (f"По этим данным нельзя оценить эффект `{t}` на `{y}`: ни один метод не применим. {fact} "
            + (f"Методам, которые подошли бы по виду данных ({names}), данных не хватает. " if names else "")
            + "Что можно сделать: собрать данные о похожих объектах без воздействия или более длинную историю "
              "до него и задать вопрос снова.")


def explain(rec: dict, lang: str, rng) -> str:
    q, gold, mode = rec["query"], rec["gold"], rec["answer_mode"]
    en = lang == "en"
    if q["question_type"] == "not_causal":
        y = rec["params"]["target"]
        return (f"This is a prediction question, not a causal one: it asks how well `{y}` can be predicted or "
                f"described, not what happens if you change something. Use a predictive model (for example gradient "
                f"boosting or regularised regression) with cross-validation, or descriptive statistics. Causal "
                f"methods are needed only if you plan to intervene on a variable and want the effect of that change."
                if en else
                f"Это вопрос о прогнозе, а не о причинности: важно, насколько хорошо можно предсказать или описать "
                f"`{y}`, а не что будет, если что-то изменить. Используйте предсказательную модель (например, "
                f"градиентный бустинг или регуляризованную регрессию) с кросс-валидацией или описательную "
                f"статистику. Причинные методы нужны, только если вы собираетесь вмешиваться и хотите знать эффект.")
    if mode == "refuse" and not q.get("hidden_confounding"):
        return _refuse_no_method(rec, lang)
    if mode == "refuse":
        t, y = q["treatment"], q["outcome"]
        f = facts(rec, lang)[-1]
        return (f"The effect of `{t}` on `{y}` cannot be identified from this data. The unmeasured confounder you "
                f"mention affects both, and there is no experiment, instrument or mediator to break that link, so "
                f"regression, matching or DML would all be biased. {f} What you can do: run a sensitivity analysis "
                f"to see how strong the hidden confounder would have to be to explain the association away; better, "
                f"run an experiment, find a valid instrument, or measure the confounder."
                if en else
                f"Эффект `{t}` на `{y}` нельзя идентифицировать по этим данным. Неизмеренный конфаундер, о котором вы "
                f"пишете, влияет на оба, а эксперимента, инструмента или медиатора, чтобы разорвать эту связь, нет: "
                f"регрессия, мэтчинг и DML будут смещены. {f} Что можно сделать: анализ чувствительности покажет, "
                f"насколько сильным должен быть скрытый конфаундер, чтобы объяснить связь; лучше провести "
                f"эксперимент, найти инструмент или измерить конфаундер.")

    ranked = sorted(gold.items(), key=lambda kv: -kv[1])
    top = [m for m, w in ranked if w >= 0.9]
    fs = facts(rec, lang)[: int(rng.integers(1, 4))]
    out = []
    if mode == "options":
        names = _join([name(m, lang) for m in top], lang)
        out.append(str(rng.choice([f"Several methods fit this data about equally well: {names}.",
                                   f"There is no single best choice here; {names} are both good options."]))
                   if en else f"Здесь подходят несколько методов примерно одинаково хорошо: {names}.")
        out += fs
        for m in top:
            p = pros(rec, m, lang)
            desc = p[0] if p else (CARDS[m].summary if en else CARDS[m].summary_ru)
            out.append(f"- {name(m, lang)}: {desc}")
        for a in top:
            for b in top:
                if a != b and b in CARDS[a].prefer_over and en:
                    out.append(f"Why {name(a, lang)} may be better than {name(b, lang)}: {CARDS[a].prefer_over[b]}.")
        out.append(("Assumptions to check: " if en else "Проверьте допущения: ")
                   + "; ".join(assumptions(top[0], lang, rng)) + ".")
        return "\n".join(out)

    best = ranked[0][0]
    out.append(str(rng.choice([f"Recommended method: {name(best, lang)}.", f"I recommend {name(best, lang)}.",
                               f"Best fit: {name(best, lang)}.", f"Use {name(best, lang)}."]))
               if en else str(rng.choice([f"Рекомендуемый метод: {name(best, lang)}.",
                                          f"Лучше всего подойдёт {name(best, lang)}."])))
    if rng.random() < 0.5:
        out.append(CARDS[best].summary if en else CARDS[best].summary_ru)
    out += fs
    p = pros(rec, best, lang)
    if p:
        out.append((f"Why it fits: {'; '.join(p[:2])}." if en else f"Почему подходит: {'; '.join(p[:2])}."))
    out += _caveats(rec, best, lang)
    alts = [m for m, w in ranked[1:] if w >= 0.5]
    if alts:
        a = alts[0]
        ap = pros(rec, a, lang)
        out.append((f"Alternative: {name(a, lang)}" + (f", since {ap[0]}." if ap else ".")) if en else
                   (f"Альтернатива: {name(a, lang)}" + (f", так как {ap[0]}." if ap else ".")))
    if (av := _avoid(rec, {best, *alts}, lang, rng)):
        out.append(av)
    if q["question_type"] == "discovery":
        out.append("Treat the resulting graph as a hypothesis and check it against domain knowledge." if en else
                   "Считайте полученный граф гипотезой и сверьте его со знанием предметной области.")
    out.append(("Assumptions to check: " if en else "Проверьте допущения: ")
               + "; ".join(assumptions(best, lang, rng)) + ".")
    return "\n".join(out)


def target(rec: dict, lang: str, rng) -> dict:
    fields = parse_fields(rec)
    return {"parse_fields": fields, "parse": parse_text(fields), "diag": diag_text(rec),
            "mode": rec["answer_mode"], "methods": rec["gold"], "explanation": explain(rec, lang, rng)}
