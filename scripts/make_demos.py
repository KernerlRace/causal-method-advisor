"""Собирает демо-датасеты для интерфейса из сгенерированных сценариев (одноразовый скрипт).

  .venv/Scripts/python.exe scripts/make_demos.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC, DST = ROOT / "data" / "scenarios" / "csv", ROOT / "data" / "demo"

DEMOS = [  # (файл, id сценария, вопрос, что ожидаем)
    ("smoking.csv", None,
     "Did Proposition 99 (the `california` law) reduce cigarette sales (`cigsale`)?",
     "Real data (Abadie et al.): one treated state, non-parallel trends → SDID / synthetic control"),
    ("marketing_ab_test.csv", "rct-10000001",
     "We ran an A/B test with free shipping. Did it increase the conversion rate?",
     "Randomized experiment → difference in means"),
    ("training_threshold.csv", "rdd-10000019",
     "Employees with an appraisal score of at least 63.8 got job training. Did the training raise productivity?",
     "Treatment assigned by a threshold → regression discontinuity"),
    ("tax_instrument.csv", "instrumental-10000084",
     "Does a higher tax rate reduce cigarette sales? Local economic conditions affect both and we can't measure "
     "them, but the neighbouring state's policy (`neighbor_policy`) shifts the tax rate without affecting sales directly.",
     "Hidden confounder + instrument → IV"),
    ("drug_rollout_panel.csv", "panel_factor-10000010",
     "The new drug was introduced in only one clinic. Did it shorten recovery days?",
     "One treated unit, many controls → SDID / synthetic control"),
    ("tutoring_staggered.csv", "panel_staggered-10000015",
     "Schools started tutoring at different times. What is the effect of tutoring on the graduation rate?",
     "Staggered adoption → SDID per cohort, naive TWFE is biased"),
    ("push_observational.csv", "backdoor_large-10000005",
     "Users who enabled push notifications seem to retain better. What is the causal effect of push notifications "
     "on retention?",
     "Large observational data, non-linear confounding → DML / doubly robust"),
    ("app_metrics_discovery.csv", "discovery_nongaussian-10000074",
     "What causes what among these app metrics?",
     "No hypothesis yet, non-Gaussian data → LiNGAM"),
    ("medicaid_timeseries.csv", "time_series-10000002",
     "After the Medicaid expansion, did GDP per capita change because of it?",
     "Single series without controls → interrupted time series"),
    ("push_unidentifiable.csv", "unidentifiable-10000003",
     "Did push notifications increase retention? Users chose to enable them, and their engagement, which we "
     "don't measure, affects both.",
     "Hidden confounder, no instrument or mediator → honest refusal + sensitivity analysis"),
    ("remote_work_frontdoor.csv", "front_door-10000016",
     "Does remote work improve productivity? Worker motivation is unobserved, but remote work affects "
     "productivity only through the skills score (`skills_score`).",
     "Hidden confounder + full mediator → front-door"),
    ("physio_continuous.csv", "continuous_treatment-10000000",
     "Как количество часов физиотерапии (`physio_hours`) влияет на `recovery_days`?",
     "Continuous treatment, linear relationships → regression (question in Russian)"),
]


def main():
    meta = []
    for fname, sid, question, expect in DEMOS:
        if sid is not None:
            df = pd.read_csv(SRC / f"{sid}.csv")
            if len(df) > 8000:  # демо держим маленькими
                df = df.sample(8000, random_state=0)
            df.to_csv(DST / fname, index=False)
        meta.append({"file": fname, "question": question, "expected": expect})
    (DST / "demos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(meta)} demos -> {DST}")


if __name__ == "__main__":
    main()
