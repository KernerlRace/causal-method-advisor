"""Вопросы, написанные вручную, а не грамматикой шаблонов: eval/handwritten_questions.jsonl.

  .venv/Scripts/python.exe -m eval.make_handwritten

Вопросы задаются к 12 демо-датасетам (data/demo). Стиль намеренно другой, чем в datagen/questions.py:
синонимы вместо имён колонок («free delivery» для free_shipping), разговорная речь и опечатки,
длинный контекст, вопросы со встречным вопросом («подойдёт ли TWFE?»), русский текст с английскими
именами колонок. Для каждого — эталонный разбор и методы, которые годятся первым номером.

Эталон разбора и методов взят из сценария, по которому сгенерирован датасет: поэтому контекст,
который нельзя увидеть в данных (рандомизация, скрытый фактор, инструмент, медиатор, порог), есть
в каждом вопросе к датасету, где он нужен, и отсутствует там, где его нет. «best» — методы с весом
≥ 0.9 в эталоне сценария (при режиме «варианты» годятся оба лучших). Для smoking.csv (реальные данные)
воздействием считается и `california`, и `after_treatment`. Вопросы про неоднородность эффекта
проверяют только разбор: эталона методов для них у этих датасетов нет.

Это ещё не «живые» вопросы от других людей — те собираются отдельно; это более строгий тест,
чем шаблонный test, потому что формулировки не взяты из грамматики, на которой училась модель.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "handwritten_questions.jsonl"

# файл → (эталонный разбор, лучшие методы, режим ответа)
GOLD = {
    "marketing_ab_test.csv": ({"type": "effect", "treatment": "free_shipping", "outcome": "conversion_rate",
                               "randomized": "yes"}, ["diff_in_means"], "recommend"),
    "training_threshold.csv": ({"type": "effect", "treatment": "job_training", "outcome": "productivity",
                                "running": "appraisal_score"}, ["rdd"], "recommend"),
    "tax_instrument.csv": ({"type": "effect", "treatment": "tax_rate", "outcome": "cigsale",
                            "instrument": "neighbor_policy", "hidden": "yes"}, ["iv"], "recommend"),
    "drug_rollout_panel.csv": ({"type": "effect", "treatment": "newDrug", "outcome": "recoveryDays"},
                               ["sdid", "synthetic_control"], "options"),
    "tutoring_staggered.csv": ({"type": "effect", "treatment": "Got Tutoring", "outcome": "Graduated Pct"},
                               ["sdid"], "recommend"),
    "push_observational.csv": ({"type": "effect", "treatment": "push_enabled", "outcome": "retained"},
                               ["dml", "doubly_robust"], "options"),
    "app_metrics_discovery.csv": ({"type": "discovery"}, ["lingam"], "recommend"),
    "medicaid_timeseries.csv": ({"type": "effect", "treatment": "Expansion", "outcome": "Gdp Per Capita"},
                                ["its"], "recommend"),
    "push_unidentifiable.csv": ({"type": "effect", "treatment": "got_push", "outcome": "retention",
                                 "randomized": "no", "hidden": "yes"}, ["sensitivity_analysis"], "refuse"),
    "remote_work_frontdoor.csv": ({"type": "effect", "treatment": "remote_work", "outcome": "productivity",
                                   "mediator": "skills_score", "hidden": "yes"}, ["front_door"], "recommend"),
    "physio_continuous.csv": ({"type": "effect", "treatment": "physio_hours", "outcome": "recovery_days"},
                              ["regression"], "recommend"),
    "smoking.csv": ({"type": "effect", "treatment": ["california", "after_treatment"], "outcome": "cigsale"},
                    ["sdid", "synthetic_control"], None),
}

QUESTIONS = {
    "marketing_ab_test.csv": [
        ("en", "We split visitors 50/50 at random last month: half saw free shipping, half didn't. Did the "
               "free shipping offer move our conversion rate, and what's the cleanest way to show it to management?"),
        ("en", "coin-flip experiment on free delivery. does it actually lift conversion or not??"),
        ("en", "Our growth team randomly assigned the free_shipping flag to users. I need a defensible estimate "
               "of how much it changed conversion_rate."),
        ("en", "Quick one: proper randomized A/B test, treatment = free shipping, metric = conversion. "
               "Which analysis do you recommend?"),
        ("en", "The experimentation platform did the randomization, so the groups should be comparable. How much "
               "higher is the conversion rate when shipping is free?"),
        ("ru", "Мы провели рандомизированный эксперимент: части пользователей случайно включили бесплатную "
               "доставку. Выросла ли от этого конверсия?"),
        ("ru", "аб тест с free_shipping, группы назначались случайно. как правильно посчитать эффект на conversion_rate"),
    ],
    "training_threshold.csv": [
        ("en", "HR only sent people to the training programme if their appraisal score was 63.8 or higher. "
               "I'd like to know whether the training itself made them more productive."),
        ("en", "Everyone scoring at least 63.8 on the yearly appraisal got job training, nobody below did. "
               "Did training boost productivity?"),
        ("en", "Training eligibility was a hard cutoff on appraisal_score. What's the effect of job_training on "
               "productivity, and how should I estimate it?"),
        ("en", "we gave training to staff above a certain appraisal mark (about 64). is the productivity bump real "
               "or just because the better employees got trained"),
        ("en", "Is there a causal effect of the training program on employee productivity? Assignment depended "
               "entirely on whether the appraisal score crossed the threshold."),
        ("ru", "Обучение получали только сотрудники с оценкой аттестации не ниже 63.8. Повысило ли обучение "
               "производительность?"),
        ("ru", "Доступ к тренингу давали по порогу appraisal_score. Как оценить влияние job_training на productivity?"),
    ],
    "tax_instrument.csv": [
        ("en", "We want to know if raising the tax rate lowers cigarette sales across counties. The problem is "
               "that local economic conditions drive both and we can't measure them. However, the neighbouring "
               "state's policy pushes our tax rate around and has no direct effect on sales."),
        ("en", "Does tax_rate causally reduce cigsale? An unobserved local economy confounds it. neighbor_policy "
               "moves the tax rate but doesn't touch sales by itself."),
        ("en", "Higher taxes, fewer cigarettes sold? There's stuff about the local economy we never recorded that "
               "affects both, but I have neighbor_policy as an exogenous shifter of the tax rate."),
        ("en", "Estimating the effect of the tax rate on cigarette sales. Something unmeasured affects both. "
               "The neighbouring state's policy affects sales only through our tax rate. What method?"),
        ("en", "What's the impact of county tax rates on cig sales? Hidden confounding for sure, but "
               "neighbor_policy shifts tax_rate and is unrelated to anything else driving sales."),
        ("ru", "Снижает ли налоговая ставка продажи сигарет? Местная экономика влияет и на то, и на другое, но мы "
               "её не измеряем. Политика соседнего штата (neighbor_policy) меняет ставку налога, а на продажи "
               "напрямую не влияет."),
        ("ru", "Между tax_rate и cigsale есть скрытый конфаундер, которого нет в данных. Зато neighbor_policy "
               "влияет на налог и больше ни на что. Как оценить эффект налога на продажи?"),
    ],
    "drug_rollout_panel.csv": [
        ("en", "Only one of our clinics switched to the new drug, the rest kept the old treatment. We have monthly "
               "recovery times for every clinic. Did the new drug actually shorten recovery?"),
        ("en", "Monthly panel of clinics. One clinic adopted newDrug at some point. How do I estimate its effect "
               "on recoveryDays?"),
        ("en", "Did patients recover faster after their clinic started using the new medication?"),
        ("en", "We rolled out the new drug in a single clinic. Is the drop in recovery days there causal, or would "
               "it have happened anyway?"),
        ("en", "What's the causal impact of the newDrug rollout on average recovery days per clinic-month?"),
        ("ru", "Новый препарат начали использовать только в одной клинике. Сократилось ли из-за него время "
               "восстановления пациентов?"),
        ("ru", "Есть помесячные данные по клиникам. Как понять, повлиял ли newDrug на recoveryDays?"),
    ],
    "tutoring_staggered.csv": [
        ("en", "Schools joined the tutoring programme in different years: some early, some late, some never. "
               "What did tutoring do to graduation rates?"),
        ("en", "Staggered rollout of tutoring across schools. I want the effect on the graduation percentage. "
               "Is a simple two-way fixed effects regression OK here?"),
        ("en", "Did getting tutoring raise the share of students who graduate?"),
        ("en", "each school started tutoring at its own time. how should i measure the impact on Graduated Pct"),
        ("en", "Estimate the effect of the tutoring program on school graduation percentages, given that adoption "
               "dates differ by school."),
        ("ru", "Школы подключались к программе репетиторства в разное время. Как она повлияла на процент выпускников?"),
        ("ru", "Got Tutoring вводили поэтапно. Есть ли эффект на Graduated Pct?"),
    ],
    "push_observational.csv": [
        ("en", "Users who turned on push notifications retain better, but they also differ in lots of ways (age, "
               "platform, past activity). What's the real effect of enabling push on retention?"),
        ("en", "We have about 8k users with a bunch of behavioural features. Does enabling push actually cause "
               "higher retention, or is it just correlation?"),
        ("en", "push_enabled vs retained: I need a causal estimate. We've logged a lot of covariates about each user."),
        ("en", "Does turning on notifications keep people around longer? Lots of covariates available, the "
               "relationships are probably messy."),
        ("en", "How much do push notifications increase retention once we account for user characteristics?"),
        ("ru", "Пользователи с включёнными пуш-уведомлениями чаще остаются. Это эффект пушей или просто другие "
               "люди? Как оценить влияние на retained?"),
        ("ru", "Влияют ли push-уведомления на удержание пользователей?"),
    ],
    "app_metrics_discovery.csv": [
        ("en", "I have a bunch of app metrics and no idea how they influence each other. Can you figure out the "
               "causal structure?"),
        ("en", "Which of these metrics drive the others? I want a causal graph, not just correlations."),
        ("en", "No specific hypothesis, I just want to know what causes what among minutes in app, friends, "
               "subscriptions, retention and the rest."),
        ("en", "Map out cause and effect between the variables in this file."),
        ("en", "Is it Minutes In App that drives Retained, or the other way around? And how do the rest fit in?"),
        ("ru", "Хочу понять, какие из метрик приложения влияют на другие. Можно восстановить причинный граф?"),
        ("ru", "Что здесь причина, а что следствие?"),
    ],
    "medicaid_timeseries.csv": [
        ("en", "We only have one series, daily GDP per capita for the state, and the Medicaid expansion kicked in "
               "partway through. Did the expansion change GDP per capita?"),
        ("en", "Did GDP per capita jump because Medicaid was expanded? There is no comparison state in the data."),
        ("en", "Time series question: effect of the Expansion on Gdp Per Capita?"),
        ("en", "How do I tell whether the Medicaid expansion moved per-capita GDP, given there's no control group?"),
        ("en", "Was the change in GDP per head after the expansion caused by it?"),
        ("ru", "У нас один временной ряд ВВП на душу населения, и в какой-то момент расширили Medicaid. "
               "Повлияло ли это на ВВП?"),
    ],
    "push_unidentifiable.csv": [
        ("en", "People decided themselves whether to get pushes, and how engaged they are (which we never "
               "tracked) affects both getting pushes and retention. Can we still estimate the effect of pushes on "
               "retention?"),
        ("en", "got_push wasn't randomized; users opted in. Their underlying engagement drives both opting in and "
               "retention, and we don't have it. Effect of got_push on retention?"),
        ("en", "This is observational: users chose push on their own. There's an unmeasured trait (engagement) "
               "that affects both. How should I estimate the impact on retention?"),
        ("en", "Not an experiment: users self-selected into receiving pushes, and some hidden factor we don't "
               "observe drives both that and retention. Is there any way to get the causal effect?"),
        ("ru", "Пользователи сами выбирали, получать ли пуши, а их вовлечённость, которую мы не измеряем, влияет "
               "и на это, и на удержание. Можно ли оценить эффект got_push на retention?"),
        ("ru", "Пуши не рандомизировали, люди подключали их сами. Скрытый фактор влияет и на пуши, и на удержание. "
               "Какой метод выбрать?"),
    ],
    "remote_work_frontdoor.csv": [
        ("en", "Does working remotely make people more productive? Motivation is a confounder we never measured. "
               "But remote work only affects productivity via the skills score."),
        ("en", "remote_work -> productivity. There's an unobserved confounder (manager support). All of the effect "
               "runs through skills_score, nothing else connects them."),
        ("en", "We can't measure worker motivation, and it drives both who works remotely and productivity. Remote "
               "work influences output exclusively by changing skills test scores. How do we get the causal effect?"),
        ("en", "Remote work's effect on productivity: hidden confounding is a concern, but the mechanism is fully "
               "mediated by skills_score. Which method fits?"),
        ("ru", "Повышает ли удалённая работа производительность? Есть ненаблюдаемый фактор, влияющий на оба. "
               "Удалёнка действует на производительность только через skills_score."),
        ("ru", "Есть скрытый конфаундер, но весь эффект remote_work на productivity идёт через навыки "
               "(skills_score). Как оценить эффект?"),
    ],
    "physio_continuous.csv": [
        ("en", "Patients get different amounts of physiotherapy. Does more physio time shorten recovery?"),
        ("en", "What's the dose-response effect of physio_hours on recovery_days? Hours vary continuously from "
               "patient to patient."),
        ("en", "How many fewer days to recover does each extra hour of physiotherapy buy us?"),
        ("en", "Is the number of physiotherapy hours causally related to recovery time? We have baseline score, "
               "sex, smoking and BMI as covariates."),
        ("en", "effect of physio hours on recovery days pls"),
        ("ru", "Как количество часов физиотерапии влияет на длительность восстановления?"),
        ("ru", "Сокращает ли физиотерапия (physio_hours) срок восстановления recovery_days?"),
    ],
    "smoking.csv": [
        ("en", "California passed Proposition 99 in 1988 and raised cigarette taxes. Did it reduce per-capita "
               "cigarette sales relative to other states?"),
        ("en", "Effect of the California tobacco control program on cigsale?"),
        ("en", "We have cigarette sales by state and year, 1970 to 2000. Only California got the law. How do we "
               "estimate its impact?"),
        ("en", "Did Prop 99 work? i.e. did cigarette sales drop in CA because of it"),
        ("en", "How much lower were cigarette sales in California after the tobacco law than they would have been "
               "without it?"),
        ("ru", "Закон Prop 99 приняли только в Калифорнии. Снизил ли он продажи сигарет?"),
    ],
}

# вопросы не про причинность: разбор = тип + целевая колонка, ответ — отказ
NOT_CAUSAL = [
    ("marketing_ab_test.csv", "en", "Build me a model that predicts conversion_rate from the other columns.", "conversion_rate"),
    ("push_observational.csv", "en", "Which user features are the best predictors of retention?", "retained"),
    ("physio_continuous.csv", "en", "What's the average recovery time for smokers vs non-smokers?", "recovery_days"),
    ("tax_instrument.csv", "en", "Forecast cigarette sales for next year for each county.", "cigsale"),
    ("training_threshold.csv", "en", "How well can we predict productivity from age, experience and department?",
     "productivity"),
    ("drug_rollout_panel.csv", "en", "Predict next month's recovery days for each clinic.", "recoveryDays"),
    ("app_metrics_discovery.csv", "en", "Which variables correlate most with Retained?", "Retained"),
    ("medicaid_timeseries.csv", "en", "Forecast GDP per capita for the next 30 days.", "Gdp Per Capita"),
    ("push_unidentifiable.csv", "ru", "Можно ли предсказать retention по остальным колонкам?", "retention"),
    ("tutoring_staggered.csv", "ru", "Какой средний процент выпускников в частных и государственных школах?",
     "Graduated Pct"),
]

# неоднородность эффекта: проверяется только разбор
HETEROGENEITY = [
    ("marketing_ab_test.csv", "en", "Free shipping was randomly assigned. For which acquisition channels does it "
     "lift conversion the most?", {"treatment": "free_shipping", "outcome": "conversion_rate", "randomized": "yes"}),
    ("push_observational.csv", "en", "Which users benefit most from push notifications in terms of retention? "
     "We want to target them.", {"treatment": "push_enabled", "outcome": "retained"}),
    ("physio_continuous.csv", "en", "Does the effect of physio hours on recovery differ between smokers and "
     "non-smokers?", {"treatment": "physio_hours", "outcome": "recovery_days"}),
]


def records() -> list[dict]:
    out = []
    for file, items in QUESTIONS.items():
        fields, best, mode = GOLD[file]
        for lang, q in items:
            out.append({"file": file, "lang": lang, "question": q, "fields": fields, "best": best, "mode": mode})
    for file, lang, q, target in NOT_CAUSAL:
        out.append({"file": file, "lang": lang, "question": q, "fields": {"type": "not_causal", "target": target},
                    "best": None, "mode": "refuse"})
    for file, lang, q, f in HETEROGENEITY:
        out.append({"file": file, "lang": lang, "question": q, "fields": {"type": "heterogeneity", **f},
                    "best": None, "mode": None})
    return [{"id": f"hw-{i:03d}", **r} for i, r in enumerate(out, 1)]


def main() -> None:
    recs = records()
    with OUT.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} questions -> {OUT.name} "
          f"(en {sum(r['lang'] == 'en' for r in recs)}, ru {sum(r['lang'] == 'ru' for r in recs)})")


if __name__ == "__main__":
    main()
