"""Темы (предметные области) и выбор имён колонок для сценариев.

Каждое понятие (Concept) — это человеческая фраза для будущих вопросов (этап 3), варианты имени
колонки (EN и иногда RU) и масштаб значений. Один и тот же сценарий оформляется в разных темах,
чтобы модель училась на смысле и профиле колонки, а не запоминала имена.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Concept:
    phrase: str
    names: tuple[str, ...]
    names_ru: tuple[str, ...] = ()
    loc: float = 0.0
    scale: float = 1.0
    kind: str = "num"  # num | int | cat | bin
    categories: tuple[str, ...] = ()


C = Concept


@dataclass(frozen=True)
class Theme:
    name: str
    unit: Concept
    times: tuple[str, ...]  # year | month | week | date | period
    treatments: tuple[Concept, ...]  # бинарные воздействия
    cont_treatments: tuple[Concept, ...]
    outcomes: tuple[Concept, ...]
    covariates: tuple[Concept, ...]
    instruments: tuple[Concept, ...]
    mediators: tuple[Concept, ...]
    running: tuple[Concept, ...]  # переменные с порогом (для RDD)
    person: Concept = field(default=C("customer", ("customer_id",)))  # id объекта в cross-section


THEMES: tuple[Theme, ...] = (
    Theme(
        "retail", C("store", ("store_id", "store", "shop_id"), ("магазин",)), ("week", "month", "year"),
        treatments=(C("new store layout", ("new_layout", "layout_change", "redesigned"), ("новая_выкладка",)),
                    C("loyalty program", ("loyalty_program", "in_loyalty", "loyalty_member")),
                    C("discount campaign", ("promo", "discount_campaign", "on_promo"), ("акция",))),
        cont_treatments=(C("discount rate", ("discount_pct", "discount_rate"), loc=10, scale=5),
                         C("shelf price", ("price", "shelf_price"), ("цена",), loc=50, scale=10)),
        outcomes=(C("revenue", ("revenue", "sales", "weekly_sales"), ("выручка",), loc=10000, scale=2500),
                  C("basket size", ("basket_size", "avg_check", "avg_basket"), loc=35, scale=8),
                  C("units sold", ("units_sold", "quantity"), loc=500, scale=120)),
        covariates=(C("store size", ("store_size_sqm", "floor_area"), ("площадь",), loc=1200, scale=300),
                    C("city population", ("city_population", "population"), loc=250000, scale=80000),
                    C("number of competitors", ("competitors", "n_competitors"), loc=4, scale=2, kind="int"),
                    C("store age", ("store_age_years", "store_age"), loc=12, scale=6),
                    C("staff count", ("staff", "n_employees"), loc=25, scale=8, kind="int"),
                    C("region", ("region", "area"), kind="cat", categories=("north", "south", "east", "west")),
                    C("customer age", ("customer_age", "age"), ("возраст",), loc=41, scale=12),
                    C("past purchases", ("past_purchases", "prior_orders"), loc=8, scale=4, kind="int"),
                    C("parking spaces", ("parking_spaces",), loc=60, scale=25, kind="int")),
        instruments=(C("distance to the distribution center", ("dist_to_dc_km", "warehouse_distance"), loc=40, scale=15),),
        mediators=(C("foot traffic", ("foot_traffic", "visitors"), loc=3000, scale=700),),
        running=(C("loyalty points", ("loyalty_points", "points_balance"), loc=500, scale=200),),
        person=C("customer", ("customer_id", "client_id")),
    ),
    Theme(
        "marketing", C("region", ("region_id", "market", "dma"), ("регион",)), ("week", "date", "month"),
        treatments=(C("email campaign", ("got_email", "email_sent", "emailed"), ("получил_письмо",)),
                    C("discount coupon", ("coupon", "got_coupon", "voucher")),
                    C("TV advertising", ("tv_campaign", "tv_on")),
                    C("free shipping offer", ("free_shipping",))),
        cont_treatments=(C("ad spend", ("ad_spend", "marketing_spend"), ("расходы_на_рекламу",), loc=5000, scale=1500),
                         C("number of ad impressions", ("impressions",), loc=20000, scale=6000)),
        outcomes=(C("conversion", ("conversions", "conversion_rate", "orders"), ("конверсия",), loc=120, scale=30),
                  C("customer spend", ("spend", "purchase_amount", "revenue"), loc=80, scale=25),
                  C("site visits", ("visits", "sessions"), loc=900, scale=200)),
        covariates=(C("customer age", ("age", "customer_age"), ("возраст",), loc=38, scale=11),
                    C("tenure in months", ("tenure_months", "months_active"), loc=20, scale=10),
                    C("previous spend", ("prev_spend", "historic_spend"), loc=300, scale=120),
                    C("device type", ("device", "platform"), kind="cat", categories=("mobile", "desktop", "tablet")),
                    C("newsletter subscriber", ("subscribed",), kind="cat", categories=("yes", "no")),
                    C("number of past orders", ("past_orders", "n_orders"), loc=5, scale=3, kind="int"),
                    C("income", ("income", "household_income"), ("доход",), loc=55000, scale=18000),
                    C("cart value", ("cart_value",), loc=60, scale=20),
                    C("channel", ("acquisition_channel",), kind="cat", categories=("search", "social", "referral", "direct"))),
        instruments=(C("random server assignment", ("server_bucket", "assignment_bucket"), kind="bin"),),
        mediators=(C("email opens", ("opened", "open_count"), loc=3, scale=1.5),),
        running=(C("engagement score", ("engagement_score", "rfm_score"), loc=50, scale=20),),
        person=C("customer", ("customer_id", "user_id")),
    ),
    Theme(
        "health", C("hospital", ("hospital_id", "hospital", "clinic"), ("больница",)), ("month", "year", "week"),
        treatments=(C("new drug", ("new_drug", "treated", "drug_a"), ("препарат",)),
                    C("telemedicine program", ("telemedicine", "telehealth")),
                    C("smoking cessation program", ("cessation_program",))),
        cont_treatments=(C("drug dosage", ("dose_mg", "dosage"), ("доза",), loc=50, scale=15),
                         C("hours of physiotherapy", ("physio_hours",), loc=10, scale=4)),
        outcomes=(C("recovery time", ("recovery_days", "days_to_recovery"), ("дни_восстановления",), loc=14, scale=4),
                  C("blood pressure", ("systolic_bp", "blood_pressure"), loc=130, scale=15),
                  C("readmissions", ("readmissions", "readmission_rate"), loc=40, scale=10)),
        covariates=(C("patient age", ("age", "patient_age"), ("возраст",), loc=55, scale=14),
                    C("body mass index", ("bmi",), loc=27, scale=4.5),
                    C("number of chronic conditions", ("chronic_conditions", "comorbidities"), loc=1.5, scale=1.2, kind="int"),
                    C("sex", ("sex", "gender"), kind="cat", categories=("female", "male")),
                    C("smoker", ("smoker",), kind="cat", categories=("yes", "no")),
                    C("baseline severity", ("severity", "baseline_score"), loc=5, scale=2),
                    C("cholesterol", ("cholesterol",), loc=200, scale=35),
                    C("insurance type", ("insurance",), kind="cat", categories=("public", "private", "none"))),
        instruments=(C("distance to the specialist clinic", ("distance_km", "dist_to_clinic"), loc=25, scale=12),),
        mediators=(C("medication adherence", ("adherence", "pills_taken_pct"), loc=70, scale=15),),
        running=(C("risk score", ("risk_score", "triage_score"), loc=50, scale=18),),
        person=C("patient", ("patient_id",)),
    ),
    Theme(
        "education", C("school", ("school_id", "school", "district"), ("школа",)), ("year", "period"),
        treatments=(C("tutoring program", ("tutoring", "got_tutoring"), ("репетитор",)),
                    C("smaller class size", ("small_class",)),
                    C("free laptop", ("laptop_program", "got_laptop"))),
        cont_treatments=(C("study hours", ("study_hours", "hours_studied"), ("часы_учёбы",), loc=10, scale=4),
                         C("years of education", ("education_years", "schooling"), loc=13, scale=2.5)),
        outcomes=(C("test score", ("test_score", "exam_score", "math_score"), ("балл",), loc=70, scale=12),
                  C("graduation rate", ("graduation_rate", "graduated_pct"), loc=82, scale=7),
                  C("earnings", ("earnings", "wage", "salary"), ("зарплата",), loc=42000, scale=12000)),
        covariates=(C("parental income", ("parent_income", "family_income"), loc=60000, scale=20000),
                    C("prior grade", ("prior_gpa", "previous_grade"), loc=3.0, scale=0.5),
                    C("student age", ("age",), ("возраст",), loc=16, scale=1.5),
                    C("gender", ("gender", "sex"), kind="cat", categories=("female", "male")),
                    C("absences", ("absences", "days_absent"), loc=6, scale=3, kind="int"),
                    C("school type", ("school_type",), kind="cat", categories=("public", "private", "charter")),
                    C("class size", ("class_size",), loc=25, scale=4, kind="int"),
                    C("parental education", ("parent_education_years",), loc=13, scale=2.5)),
        instruments=(C("distance to college", ("distance_to_college", "college_proximity"), loc=20, scale=10),
                     C("quarter of birth", ("birth_quarter",), loc=2.5, scale=1.1, kind="int")),
        mediators=(C("homework completion", ("homework_done_pct",), loc=75, scale=12),),
        running=(C("entrance exam score", ("entrance_score", "admission_score"), loc=60, scale=15),),
        person=C("student", ("student_id",)),
    ),
    Theme(
        "labor", C("company", ("company_id", "firm", "employer"), ("компания",)), ("year", "month"),
        treatments=(C("job training program", ("training", "job_training"), ("обучение",)),
                    C("remote work policy", ("remote_work", "wfh_policy")),
                    C("union membership", ("union_member",))),
        cont_treatments=(C("training hours", ("training_hours",), loc=20, scale=8),
                         C("bonus size", ("bonus", "bonus_amount"), ("премия",), loc=3000, scale=1000)),
        outcomes=(C("salary", ("salary", "wage", "annual_pay"), ("зарплата",), loc=52000, scale=14000),
                  C("productivity", ("productivity", "output_per_hour"), loc=100, scale=18),
                  C("employee turnover", ("turnover_rate", "attrition"), ("текучесть",), loc=15, scale=5)),
        covariates=(C("years of experience", ("experience_years", "experience"), ("стаж",), loc=8, scale=5),
                    C("employee age", ("age",), ("возраст",), loc=37, scale=10),
                    C("education level", ("education",), kind="cat", categories=("school", "bachelor", "master", "phd")),
                    C("team size", ("team_size",), loc=9, scale=4, kind="int"),
                    C("department", ("department", "dept"), kind="cat", categories=("sales", "it", "ops", "finance")),
                    C("performance rating", ("last_rating", "performance_score"), loc=3.5, scale=0.8),
                    C("commute time", ("commute_minutes",), loc=35, scale=15),
                    C("company size", ("company_size",), loc=500, scale=250, kind="int")),
        instruments=(C("lottery for the training slot", ("training_lottery", "lottery_win"), kind="bin"),),
        mediators=(C("skills test score", ("skills_score",), loc=60, scale=12),),
        running=(C("performance score", ("perf_score", "appraisal_score"), loc=70, scale=12),),
        person=C("employee", ("employee_id", "worker_id")),
    ),
    Theme(
        "policy", C("state", ("state", "state_id", "country"), ("штат",)), ("year",),
        treatments=(C("tobacco tax law", ("tax_law", "policy", "law_passed"), ("закон",)),
                    C("minimum wage increase", ("min_wage_increase", "wage_policy")),
                    C("lockdown", ("lockdown",)),
                    C("medicaid expansion", ("expansion", "medicaid_expansion"))),
        cont_treatments=(C("tax rate", ("tax_rate",), ("налог",), loc=8, scale=3),
                         C("police spending per capita", ("police_spending",), loc=300, scale=90)),
        outcomes=(C("cigarette sales per capita", ("cigsale", "cig_sales", "packs_per_capita"), loc=120, scale=30),
                  C("employment rate", ("employment_rate", "employment"), ("занятость",), loc=62, scale=5),
                  C("crime rate", ("crime_rate", "crimes_per_100k"), loc=400, scale=120),
                  C("GDP per capita", ("gdp_per_capita", "gdp_pc"), loc=50000, scale=9000)),
        covariates=(C("median income", ("median_income", "lnincome"), loc=48000, scale=9000),
                    C("unemployment rate", ("unemployment",), ("безработица",), loc=6, scale=2),
                    C("share of young adults", ("age15to24", "young_share"), loc=0.17, scale=0.02),
                    C("population", ("population",), loc=5e6, scale=2e6),
                    C("urban share", ("urban_share",), loc=0.7, scale=0.12),
                    C("retail price", ("retprice", "avg_price"), loc=100, scale=30),
                    C("beer consumption", ("beer",), loc=24, scale=4)),
        instruments=(C("neighbouring state's policy", ("neighbor_policy",), kind="bin"),),
        mediators=(C("average price", ("avg_price",), loc=100, scale=25),),
        running=(C("poverty rate", ("poverty_rate",), loc=14, scale=4),),
        person=C("county", ("county_id", "fips")),
    ),
    Theme(
        "tech", C("country", ("country", "market", "country_code"), ("страна",)), ("date", "week", "month"),
        treatments=(C("new onboarding flow", ("new_onboarding", "variant_b", "treatment_group"), ("новый_онбординг",)),
                    C("push notifications", ("push_enabled", "got_push")),
                    C("premium trial", ("premium_trial", "trial"))),
        cont_treatments=(C("number of notifications", ("n_notifications", "push_count"), loc=5, scale=2),
                         C("app load time", ("load_time_ms",), loc=800, scale=200)),
        outcomes=(C("retention", ("retention_d30", "retained", "retention"), ("удержание",), loc=40, scale=10),
                  C("time in app", ("minutes_in_app", "session_minutes"), loc=25, scale=8),
                  C("subscriptions", ("subscriptions", "paid_conversions"), loc=60, scale=15)),
        covariates=(C("user age", ("age", "user_age"), ("возраст",), loc=29, scale=8),
                    C("days since signup", ("days_since_signup", "account_age"), loc=120, scale=80),
                    C("operating system", ("os", "platform"), kind="cat", categories=("ios", "android", "web")),
                    C("sessions last week", ("sessions_7d", "prev_sessions"), loc=6, scale=3, kind="int"),
                    C("friends on platform", ("n_friends", "friends"), loc=15, scale=8, kind="int"),
                    C("country tier", ("country_tier",), kind="cat", categories=("tier1", "tier2", "tier3")),
                    C("prior purchases", ("prior_purchases",), loc=2, scale=1.5, kind="int"),
                    C("device price", ("device_price",), loc=500, scale=200)),
        instruments=(C("random rollout bucket", ("rollout_bucket", "hash_bucket"), kind="bin"),),
        mediators=(C("feature usage", ("feature_usage", "feature_clicks"), loc=10, scale=4),),
        running=(C("activity score", ("activity_score",), loc=50, scale=20),),
        person=C("user", ("user_id", "account_id")),
    ),
)

ANON_COVARIATE_STYLES = ("x{}", "X{}", "feature_{}", "var{}", "col_{}", "f{}")


def _style(name: str, style: str) -> str:
    parts = re.split(r"_+", name)
    if style == "camel":
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    if style == "title":
        return " ".join(p.capitalize() for p in parts)
    return name


class Namer:
    """Выдаёт уникальные имена колонок в одном стиле для одного датасета."""

    def __init__(self, rng: np.random.Generator, theme: Theme | None = None, anonymous: bool | None = None,
                 russian: bool | None = None):
        self.rng = rng
        self.theme = theme or THEMES[rng.integers(len(THEMES))]
        self.anonymous = bool(rng.random() < 0.12) if anonymous is None else anonymous
        self.russian = (not self.anonymous and rng.random() < 0.10) if russian is None else russian
        self.style = rng.choice(["snake", "snake", "snake", "camel", "title"])
        self.used: set[str] = set()
        self._anon_cov = rng.choice(ANON_COVARIATE_STYLES)
        self._anon_i = 1
        self.concepts: dict[str, str] = {}  # колонка -> фраза для вопросов
        self.used_phrases: set[str] = set()
        self._generic_i = 0

    def _unique(self, base: str) -> str:
        name, k = base, 2
        while name.lower() in self.used:
            name, k = f"{base}_{k}", k + 1
        self.used.add(name.lower())
        return name

    def pick(self, concept: Concept, anon_names: tuple[str, ...], role: str) -> str:
        if self.anonymous:
            base = str(self.rng.choice(anon_names))
            phrase = {"treatment": "the treatment", "outcome": "the outcome"}.get(role, f"variable {base}")
        else:
            pool = concept.names_ru if (self.russian and concept.names_ru) else concept.names
            base = _style(str(self.rng.choice(pool)), self.style)
            phrase = concept.phrase
        name = self._unique(base)
        self.concepts[name] = phrase
        self.used_phrases.add(concept.phrase)
        return name

    def covariate(self, concept: Concept) -> str:
        if self.anonymous:
            name = self._unique(self._anon_cov.format(self._anon_i))
            self._anon_i += 1
            self.concepts[name] = f"variable {name}"
            self.used_phrases.add(concept.phrase)
            return name
        return self.pick(concept, (), "covariate")

    def sample(self, pool: tuple[Concept, ...], k: int = 1) -> list[Concept]:
        """k ещё не использованных понятий; если темы не хватает — добираем обобщёнными."""
        free = [c for c in pool if c.phrase not in self.used_phrases]
        idx = self.rng.choice(len(free), size=min(k, len(free)), replace=False) if free else []
        out = [free[i] for i in idx]
        while len(out) < k:
            self._generic_i += 1
            out.append(generic_concept(self._generic_i, self.rng))
        return out


GENERIC_KINDS = (("survey item {}", ("survey_q{}", "q{}_score"), 3.0, 1.0),
                 ("metric {}", ("metric_{}", "kpi_{}"), 100.0, 25.0),
                 ("index {}", ("index_{}", "idx{}"), 50.0, 15.0))


def generic_concept(i: int, rng) -> Concept:
    phrase, names, loc, scale = GENERIC_KINDS[int(rng.integers(len(GENERIC_KINDS)))]
    return Concept(phrase.format(i), tuple(n.format(i) for n in names), loc=loc, scale=scale)


def realize(concept: Concept, z: np.ndarray, rng: np.random.Generator):
    """Стандартизованный латентный сигнал z -> значения в масштабе понятия."""
    if concept.kind == "bin":
        return (z > 0).astype(int)
    if concept.kind == "cat":
        cats = np.array(concept.categories)
        q = np.quantile(z, np.linspace(0, 1, len(cats) + 1)[1:-1])
        return cats[np.searchsorted(q, z)]
    v = concept.loc + concept.scale * z
    if concept.kind == "int":
        return np.maximum(np.round(v), 0).astype(int)
    if concept.loc > 0:  # положительные по смыслу величины (доход, население) не бывают < 0
        v = np.maximum(v, 0)
    return v
