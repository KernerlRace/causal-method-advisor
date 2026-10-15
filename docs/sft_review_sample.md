# SFT review sample

## 1. backdoor_large (en, mode=options)

**Question:** I'm analysing our retail data. What happened to basket size (`avg_check`) because of loyalty program (loyalty_program)? Note that loyalty program (loyalty_program) was not randomized: customers chose it themselves.

```
rows=13636 cols=27 structure=cross_section
col client_id integer uniq=13636 range=[1000..1.464e+04]
col n_employees integer uniq=55 range=[0..56]
col parking_spaces integer uniq=143 range=[0..145] missing=6%
col age numeric uniq=4649 range=[0..82.01]
col city_population numeric uniq=13624 range=[0..5.876e+05]
col floor_area numeric uniq=13565 range=[83.34..2442]
col past_purchases integer uniq=23 range=[0..22]
col area categorical uniq=4 values=north|east|south|west
col store_age numeric uniq=9876 range=[0..34.58]
col competitors integer uniq=13 range=[0..12]
col index_1 numeric uniq=12039 range=[0..103]
col survey_q2 numeric uniq=581 range=[0..6.52]
col q3_score numeric uniq=587 range=[0..7.21]
col kpi_4 numeric uniq=7322 range=[0..193.8]
col index_5 numeric uniq=5400 range=[0..119.1]
col q6_score numeric uniq=4047 range=[0..6.902]
col idx7 numeric uniq=5383 range=[0..105.4]
col survey_q8 numeric uniq=4064 range=[0..7.087]
col survey_q9 numeric uniq=588 range=[0..6.66]
col kpi_10 numeric uniq=7293 range=[0..188.7]
col idx11 numeric uniq=12038 range=[0..115.6]
col survey_q12 numeric uniq=4065 range=[0..7.175]
col idx13 numeric uniq=12063 range=[0..111.3]
col index_14 numeric uniq=12063 range=[0..110.1]
col loyalty_program binary uniq=2 range=[0..1] values=0|1
col avg_check numeric uniq=10789 range=[0.16..84.38]
col q15_score numeric uniq=590 range=[0..7.64]
```

**Parse:** `<T:effect> treatment=`loyalty_program` outcome=`avg_check` randomized=`no``

```
size rows=13636 covariates=25
overlap auc=0.775 extreme=0.0154 maxsmd=0.939
nonlinearity r2_linear=0.315 r2_boosting=0.752
flags good_overlap high_dim imbalance large_sample nonlinear
allowed dml doubly_robust causal_forest psm ipw sensitivity_analysis
warning matching regression
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

Several methods fit this data about equally well: Double machine learning and Doubly robust estimation (AIPW / DR-learner).
The outcome depends on the covariates non-linearly: boosting reaches R²=0.752 vs 0.315 for a linear model.
- Double machine learning: flexible ML models capture the non-linear relationships found in the data
- Doubly robust estimation (AIPW / DR-learner): flexible models in both parts capture the non-linear relationships
Assumptions to check: the ML models predict outcome and treatment reasonably well; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 2. backdoor_small (ru, mode=recommend)

**Question:** Есть данные, нужен совет. Сработал ли `expansion`? Хочу понять, как он повлиля на cigsale. Спасибо!

```
rows=122 cols=6 structure=cross_section
col cigsale numeric uniq=121 range=[35.22..212.3]
col population numeric uniq=122 range=[1.154e+06..8.919e+06]
col expansion binary uniq=2 values=yes|no
col urban_share numeric uniq=49 range=[0.33..1.03]
col unemployment numeric uniq=117 range=[1.49..10.3]
col retprice numeric uniq=121 range=[19.84..155.9]
```

**Parse:** `<T:effect> treatment=`expansion` outcome=`cigsale``

```
size rows=122 covariates=4
overlap auc=0.665 extreme=0 maxsmd=0.685
nonlinearity r2_linear=0.0739 r2_boosting=0.154
flags good_overlap imbalance nonlinear small_sample
allowed sensitivity_analysis
warning ipw matching psm regression diff_in_means
```

**Methods:** {'regression': 0.6, 'psm': 0.6, 'ipw': 0.5}

**Explanation:**

Рекомендуемый метод: Линейная регрессия с контролем конфаундеров.
Регрессия исхода на воздействие и набор конфаундеров, закрывающий back-door пути.
Исход зависит от ковариат нелинейно: бустинг даёт R²=0.154, линейная модель 0.0739.
Ковариаты предсказывают воздействие с AUC=0.665; у 0% объектов экстремальный propensity score.
В данных 122 строк.
Почему подходит: данных мало.
Альтернатива: Мэтчинг по propensity score, так как группы хорошо пересекаются по ковариатам.
Разность средних в эксперименте здесь хуже: группы заметно различаются по ковариатам.
Проверьте допущения: все конфаундеры наблюдаемы и включены в модель; в модель не добавлены медиаторы и коллайдеры.

## 3. backdoor_targeted (en, mode=options)

**Question:** We have data on students. Does `d` lead to higher `y`? Which method should I use?

```
rows=1071 cols=7 structure=cross_section
col X1 numeric uniq=807 range=[1.396..4.636]
col d binary uniq=2 values=no|yes
col X3 integer uniq=26 range=[12..39]
col X2 numeric uniq=650 range=[5.37..21.22]
col y numeric uniq=1071 range=[8043..9.169e+04]
col X4 binary uniq=2 values=female|male
col row_id integer uniq=1071 range=[1000..2070]
```

**Parse:** `<T:effect> treatment=`d` outcome=`y``

```
size rows=1071 covariates=5
overlap auc=0.964 extreme=0.523 maxsmd=1.31
nonlinearity r2_linear=0.548 r2_boosting=0.769
flags imbalance nonlinear poor_overlap
allowed sensitivity_analysis
warning causal_forest dml doubly_robust diff_in_means ipw matching psm regression
```

**Methods:** {'doubly_robust': 1.0, 'psm': 0.9, 'matching': 0.6, 'regression': 0.3, 'ipw': 0.2}

**Explanation:**

The groups barely overlap, so there is no single safe answer: the right method depends on which effect you need.
Covariates predict the treatment with AUC=0.964; 52% of units have extreme propensity scores. For these units there is no comparable unit in the other group, so the effect for everyone can only be extrapolated.
Options:
- Doubly robust estimation (AIPW / DR-learner) after trimming the units with extreme propensity scores: the effect for units that could have gone either way, not for everyone.
- Propensity score matching with a caliper: units without a close match are dropped, so the effect is for the matched units, usually the treated.
Avoid: Linear regression with an adjustment set, because the outcome depends on the covariates non-linearly, so a linear model is misspecified.
Whichever you choose, report how many units are dropped and how they differ from the rest.
Assumptions to check: all confounders are observed; overlap between groups.

## 4. continuous_treatment (en, mode=recommend)

**Question:** Did blood pressure respond to the hours of physiotherapy? Thanks

```
rows=3915 cols=8 structure=cross_section
col patient_id integer uniq=3915 range=[1000..4914]
col smoker binary uniq=2 values=yes|no
col gender binary uniq=2 values=female|male
col insurance categorical uniq=3 values=public|none|private
col physio_hours numeric uniq=3528 range=[0.08..35]
col systolic_bp numeric uniq=3794 range=[87.4..179.9]
col bmi numeric uniq=1619 range=[10.55..44.36]
col cholesterol numeric uniq=3860 range=[81.17..334.2]
```

**Parse:** `<T:effect> treatment=`physio_hours` outcome=`systolic_bp``

```
size rows=3915 covariates=6
nonlinearity r2_linear=0.801 r2_boosting=0.759
flags linear_ok
allowed regression causal_forest dml sensitivity_analysis
warning -
```

**Methods:** {'regression': 1.0, 'dml': 0.8}

**Explanation:**

Best fit: Linear regression with an adjustment set.
Regress the outcome on the treatment plus a set of confounders that blocks all back-door paths.
A linear model fits the outcome about as well as boosting (R²=0.801 vs 0.759).
The data has 3915 rows.
Why it fits: relationships look linear, so a linear model is adequate.
Alternative: Double machine learning.
Assumptions to check: no mediators or colliders are included as controls; the relationship is approximately linear (or correctly specified).

## 5. discovery_gaussian (en, mode=options)

**Question:** I have a CSV with 5039 rows. Which of these variables are causes of Severity? What approach would you recommend?

```
rows=5039 cols=4 structure=cross_section
col Readmissions numeric uniq=2796 range=[0.2..81.79]
col Bmi numeric uniq=1740 range=[10.58..46.82]
col Severity numeric uniq=945 range=[0.04..15.38]
col Age numeric uniq=3243 range=[4.37..108.2]
```

**Parse:** `<T:discovery>`

```
size rows=5039 covariates=4
normality nongaussian=0
flags gaussian_data large_sample
allowed ges notears pc fci
warning lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

Several methods fit this data about equally well: PC algorithm and Greedy equivalence search.
0% of the continuous variables are clearly non-Gaussian.
The data has 5039 rows.
- PC algorithm: independence tests are reliable on large samples
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: faithfulness (independences in data come only from the graph); independent observations.

## 6. discovery_hidden (en, mode=recommend)

**Question:** Quick question: I don't have a hypothesis yet. What causes what among graduation rate, parental education and prior grade? Not all relevant factors are in the data.

```
rows=906 cols=6 structure=cross_section
col Wage numeric uniq=906 range=[2754..8.837e+04]
col Age numeric uniq=819 range=[10.72..20.89]
col Previous Grade numeric uniq=718 range=[1.567..4.442]
col Graduated Pct numeric uniq=761 range=[63.4..101.7]
col Parent Education Years numeric uniq=861 range=[4.422..19.89]
col Family Income numeric uniq=906 range=[2.548e+04..9.363e+04]
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=906 covariates=6
normality nongaussian=1
flags non_gaussian
allowed -
warning fci lingam ges notears pc
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

Recommended method: Fast causal inference (FCI).
100% of the continuous variables are clearly non-Gaussian.
The data has 906 rows.
Why it fits: designed for data with possible hidden confounders.
Works best with at least 1000 rows, has 906.
Avoid: PC algorithm, because PC assumes no hidden confounders, consider FCI.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 7. discovery_nongaussian (en, mode=recommend)

**Question:** Can you find the causal graph behind this data?

```
rows=4132 cols=6 structure=cross_section
col Age numeric uniq=2961 range=[12.27..20.14]
col Family Income numeric uniq=4132 range=[1.377e+04..1.074e+05]
col Previous Grade numeric uniq=281 range=[1.61..5.09]
col Wage numeric uniq=4131 range=[9905..7.488e+04]
col Graduated Pct numeric uniq=1979 range=[70.18..94.25]
col Exam Score numeric uniq=3940 range=[36.48..105.4]
```

**Parse:** `<T:discovery>`

```
size rows=4132 covariates=6
normality nongaussian=1
flags non_gaussian
allowed lingam fci ges notears pc
warning -
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

Best fit: LiNGAM.
Functional discovery that uses non-Gaussian noise in linear relationships to recover the full direction of every edge.
100% of the continuous variables are clearly non-Gaussian.
The data has 4132 rows.
Why it fits: non-Gaussian data lets LiNGAM orient every edge.
Alternative: PC algorithm.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: relationships are linear; noise terms are non-Gaussian.

## 8. front_door (en, mode=recommend)

**Question:** Does tv_on lead to higher site visits? The whole effect of tv_on on site visits goes through email opens. We don't observe brand affinity, which likely influences who gets tv_on and also site visits. Which method should I use?

```
rows=3353 cols=5 structure=cross_section
col opened numeric uniq=758 range=[0.03..12.19]
col visits numeric uniq=3347 range=[35.73..1701]
col tv_on binary uniq=2 values=no|yes
col past_orders integer uniq=17 range=[0..16]
col customer_id integer uniq=3353 range=[1000..4352]
```

**Parse:** `<T:effect> treatment=`tv_on` outcome=`visits` mediator=`opened` hidden=`yes``

```
size rows=3353 covariates=2
overlap auc=0.5 extreme=0 maxsmd=0.0557
nonlinearity r2_linear=0.086 r2_boosting=-0.0296
flags good_overlap linear_ok
allowed front_door sensitivity_analysis
warning diff_in_means ipw matching psm regression causal_forest dml doubly_robust
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

I recommend Front-door adjustment.
A linear model fits the outcome about as well as boosting (R²=0.086 vs -0.0296).
Covariates predict the treatment with AUC=0.5; 0% of units have extreme propensity scores.
The data has 3353 rows.
Why it fits: front-door works where back-door adjustment is impossible.
Avoid: Linear regression with an adjustment set, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: the mediator captures the entire effect of the treatment on the outcome; all back-door paths from mediator to outcome are blocked by the treatment.

## 9. instrumental (en, mode=recommend)

**Question:** I have a CSV with 3786 rows. Does the number of notifications affect time in app (`session_minutes`)? We don't obserev tech savviness, which likely influences who gets the number of notifications and also time in app (`session_minutes`). Random rollout bucket influences the number of notifications and is unrelated to everything else that affects time in app (`session_minutes`). What's the right methodology here?

```
rows=3786 cols=5 structure=cross_section
col age numeric uniq=2140 range=[0..55.6]
col n_friends integer uniq=43 range=[0..44] missing=7%
col n_notifications numeric uniq=1269 range=[0.04..23.82]
col rollout_bucket binary uniq=2 range=[0..1] values=1|0
col session_minutes numeric uniq=3559 range=[0.16..57.08]
```

**Parse:** `<T:effect> treatment=`n_notifications` outcome=`session_minutes` instrument=`rollout_bucket` hidden=`yes``

```
size rows=3786 covariates=2
firststage F=741.0
nonlinearity r2_linear=0.168 r2_boosting=0.0203
flags linear_ok
allowed iv sensitivity_analysis
warning regression causal_forest dml
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Best fit: Instrumental variables (2SLS).
Use a variable that shifts the treatment but affects the outcome only through it, so the effect is identified even with unobserved confounders.
The instrument is strong: first-stage F=741.0.
A linear model fits the outcome about as well as boosting (R²=0.168 vs 0.0203).
Why it fits: IV is one of the few ways to handle unobserved confounders.
Not recommended: Causal forest (CausalForestDML), because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: relevance (the instrument really moves the treatment); monotonicity, for a LATE interpretation.

## 10. not_causal (en, mode=refuse)

**Question:** Hello, Which features are most predictive of time in app?

```
rows=407 cols=5 structure=cross_section
col n_friends integer uniq=37 range=[0..40] missing=8%
col account_age numeric uniq=354 range=[0..388.2] missing=7%
col sessions_7d integer uniq=17 range=[0..18]
col got_push binary uniq=2 range=[0..1] values=1|0
col minutes_in_app numeric uniq=378 range=[7.53..49.12]
```

**Parse:** `<T:not_causal> target=`minutes_in_app``

```
size rows=407 covariates=5
flags -
allowed -
warning -
```

**Methods:** {}

**Explanation:**

This is a prediction question, not a causal one: it asks how well `minutes_in_app` can be predicted or described, not what happens if you change something. Use a predictive model (for example gradient boosting or regularised regression) with cross-validation, or descriptive statistics. Causal methods are needed only if you plan to intervene on a variable and want the effect of that change.

## 11. panel_factor (en, mode=recommend)

**Question:** the job training program -> `annualPay`: is there an effect? any advice?

```
rows=770 cols=6 structure=panel
panel unit=employer(35) time=period(22) balanced
col department categorical uniq=4 values=sales|finance|it|ops
col annualPay numeric uniq=770 range=[280..1.144e+05]
col employer integer uniq=35 range=[1..35]
col jobTraining binary uniq=2 values=False|True
col education categorical uniq=4 values=school|phd|master|bachelor
col period datetime uniq=22 values=2005-09-01|2007-06-01
design indicator(jobTraining) treated=2 controls=33 pre=18 post=4 block
```

**Parse:** `<T:effect> treatment=`jobTraining` outcome=`annualPay``

```
size rows=770 covariates=2 treated=2 controls=33 pre=18 post=4
pretrend drift=3.02e+04 rel=1.64 p=3.71e-11
scfit r2=-0.0669 donors=33 outside=0
flags has_controls policy_design pretrend_violated sc_poor_fit
allowed sdid sensitivity_analysis
warning synthetic_control did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.4, 'did': 0.1}

**Explanation:**

Recommended method: Synthetic difference-in-differences.
Before the treatment the gap between treated and control units drifted by 3.02e+04 (164% of the outcome's SD, p=3.71e-11), so the trends are not parallel.
Why it fits: unit and time weights repair non-parallel trends.
Avoid: Synthetic control, because the synthetic control does not reproduce the pre-treatment path.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 12. panel_parallel (en, mode=recommend)

**Question:** I'm analysing our labor data. Should we roll out the remote work policy? Does it move salary (wage)? What's the right methodology here?

```
rows=2800 cols=6 structure=panel
panel unit=companyId(100) time=yr(28) balanced
col companyId categorical uniq=100 values=company_1|company_2|company_3|company_4|company_5
col yr integer uniq=28 range=[1990..2017]
col wage numeric uniq=2800 range=[280..1.189e+05]
col wfhPolicy binary uniq=2 values=no|yes
col lastRating numeric uniq=400 range=[0.72..6.56]
col education categorical uniq=4 values=school|bachelor|phd|master
design indicator(wfhPolicy) treated=48 controls=52 pre=19 post=9 block
```

**Parse:** `<T:effect> treatment=`wfhPolicy` outcome=`wage``

```
size rows=2800 covariates=2 treated=48 controls=52 pre=19 post=9
pretrend drift=423.3 rel=0.0207 p=0.205
scfit r2=0.768 donors=52 outside=0.0526
flags has_controls many_treated_units policy_design pretrend_ok
allowed did sdid sensitivity_analysis
warning dml its regression synthetic_control
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

Best fit: Difference-in-differences.
Pre-treatment trends look parallel (drift 423.3, p=0.205).
A synthetic control reproduces the treated path before the treatment well (R²=0.768).
There are 48 treated units and 52 control units, 19 periods before and 9 after the treatment.
Why it fits: pre-treatment trends of the groups are parallel; many treated units make the estimate precise.
Alternative: Synthetic difference-in-differences.
Not recommended: Synthetic control, because with many treated units, DiD or SDID is usually more efficient.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 13. panel_staggered (en, mode=recommend)

**Question:** hello, i'd like to evaluate `D`. did it change y? what approach would you recommend?

```
rows=2140 cols=4 structure=panel
panel unit=unit(107) time=t(20) balanced
col unit integer uniq=107 range=[1..107]
col t integer uniq=20 range=[1970..1989]
col Y numeric uniq=2137 range=[2.4..1060]
col D binary uniq=2 values=False|True
design indicator(D) treated=69 controls=38 pre=6 post=14 staggered
```

**Parse:** `<T:effect> treatment=`D` outcome=`Y``

```
size rows=2140 covariates=0 treated=69 controls=38 pre=6 post=14
pretrend drift=0.0616 rel=0.000529 p=0.992
scfit r2=1 donors=33 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit staggered
allowed sdid sensitivity_analysis
warning did synthetic_control dml regression
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Best fit: Synthetic difference-in-differences.
Weighted DiD with unit weights (like synthetic control, but only trends must match, not levels) and time weights that focus on pre-periods most similar to the post-period.
Pre-treatment trends look parallel (drift 0.0616, p=0.992).
Why it fits: staggered adoption is handled by running SDID per adoption cohort and averaging.
Not recommended: Difference-in-differences, because naive two-way fixed effects is biased when units are treated at different times.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 14. rct (en, mode=recommend)

**Question:** Hello, Does smaller class size lead to higher graduation rate? Assignment to smaller class size was random

```
rows=571 cols=8 structure=cross_section
col gender binary uniq=2 values=female|male
col parent_education_years numeric uniq=428 range=[3.65..20.46] missing=1%
col small_class binary uniq=2 values=False|True
col age numeric uniq=536 range=[10.49..20.18]
col graduation_rate numeric uniq=509 range=[58.83..99.5]
col class_size integer uniq=25 range=[13..38]
col prior_gpa numeric uniq=475 range=[1.21..4.484]
col parent_income numeric uniq=542 range=[0..1.198e+05] missing=5%
```

**Parse:** `<T:effect> treatment=`small_class` outcome=`graduation_rate` randomized=`yes``

```
size rows=571 covariates=6
overlap auc=0.508 extreme=0 maxsmd=0.173
nonlinearity r2_linear=0.242 r2_boosting=0.0831
flags good_overlap linear_ok randomization_plausible
allowed regression diff_in_means ipw matching psm sensitivity_analysis
warning causal_forest dml doubly_robust
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

Best fit: Difference in means (randomized experiment).
A linear model fits the outcome about as well as boosting (R²=0.242 vs 0.0831).
Why it fits: covariates are balanced, so randomization looks credible.
Alternative: Linear regression with an adjustment set, since relationships look linear, so a linear model is adequate.
Assumptions to check: SUTVA (one unit's treatment does not affect another unit's outcome); no differential attrition between groups.

## 15. rct_heterogeneous (en, mode=options)

**Question:** Hi! Which employees respond to the union membership? We want to personalise it. It was a randomized A/B test.

```
rows=2818 cols=7 structure=cross_section
col age numeric uniq=1964 range=[4..66.61]
col dept categorical uniq=4 values=sales|finance|ops|it
col last_rating numeric uniq=381 range=[1.02..6.18]
col company_size integer uniq=920 range=[0..1307]
col experience numeric uniq=2429 range=[0..23.53]
col union_member binary uniq=2 values=no|yes
col productivity numeric uniq=2271 range=[36.06..157.2]
```

**Parse:** `<T:heterogeneity> treatment=`union_member` outcome=`productivity` randomized=`yes``

```
size rows=2818 covariates=5
overlap auc=0.532 extreme=0 maxsmd=0.103
nonlinearity r2_linear=0.458 r2_boosting=0.425
flags good_overlap linear_ok randomization_plausible
allowed meta_learners dml doubly_robust
warning causal_forest
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

There is no single best choice here; Meta-learners (S-, T-, X-learner) and Causal forest (CausalForestDML) are both good options.
A linear model fits the outcome about as well as boosting (R²=0.458 vs 0.425).
Covariates predict the treatment with AUC=0.532; 0% of units have extreme propensity scores.
The data has 2818 rows.
- Meta-learners (S-, T-, X-learner): on experimental data meta-learners give an honest estimate of heterogeneity
- Causal forest (CausalForestDML): An ensemble of causal trees that split the sample so that the effect is homogeneous within leaves.
Why Causal forest (CausalForestDML) may be better than Meta-learners (S-, T-, X-learner): provides confidence intervals for CATE.
Assumptions to check: no unobserved confounders (holds automatically in an experiment); SUTVA.

## 16. rdd (ru, mode=recommend)

**Question:** Привет! Стоит ли раскатывать `telehealth`? Влияет ли он на `readmissions`? Как это правильно проанализировать?

```
rows=9324 cols=4 structure=cross_section
col risk_score numeric uniq=5119 range=[0.36..136.7]
col readmissions numeric uniq=8066 range=[0.2..156]
col telehealth binary uniq=2 range=[0..1] values=0|1
col insurance categorical uniq=3 values=private|none|public
```

**Parse:** `<T:effect> treatment=`telehealth` outcome=`readmissions``

```
size rows=9324 covariates=2
overlap auc=1 extreme=0.864 maxsmd=2.65
threshold var=risk_score cutoff=71.2 acc=0.995 jump=0.969
nonlinearity r2_linear=0.689 r2_boosting=0.684
flags imbalance large_sample linear_ok poor_overlap running_variable_found sharp_cutoff
allowed rdd sensitivity_analysis
warning causal_forest dml doubly_robust regression diff_in_means ipw matching psm
```

**Methods:** {'rdd': 1.0}

**Explanation:**

Лучше всего подойдёт Разрывный регрессионный дизайн (RDD).
Воздействие включается при `risk_score` >= 71.2: это правило объясняет 100% назначений.
Почему подходит: воздействие назначено по порогу непрерывной переменной; порог определяет воздействие почти точно.
Мэтчинг по ковариатам здесь хуже: у части объектов вероятность воздействия почти 0 или почти 1.
Проверьте допущения: объекты не могут точно подогнать значение под порог; эффект оценивается только около порога.

## 17. time_series (en, mode=recommend)

**Question:** Hey, I need to know whether telemedicine program (Telemedicine) caused a change in Systolic Bp.

```
rows=65 cols=5 structure=time_series
time_series time=month
col Systolic Bp numeric uniq=65 range=[125.5..147.1]
col Smoker binary uniq=2 values=yes|no
col Telemedicine binary uniq=2 range=[0..1] values=0|1
col month datetime uniq=65 values=2007-01-01|2012-05-01
col Age numeric uniq=58 range=[32.06..87.4] missing=11%
```

**Parse:** `<T:effect> treatment=`Telemedicine` outcome=`Systolic Bp``

```
size rows=65 covariates=2 pre=46 post=19
flags no_controls
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Recommended method: Interrupted time series / CausalImpact.
Model the pre-treatment dynamics of the series (optionally with covariate series) and forecast the counterfactual after the intervention.
This is a single series with no control group: 46 periods before the intervention and 19 after.
Why it fits: there is no control group, so a time-series counterfactual is the main option.
Assumptions to check: no other event coincides with the intervention; covariate series (if used) are not affected by the treatment.

## 18. unidentifiable (ru, mode=refuse)

**Question:** есть данные, нужен совет. как оценить влияние `min_wage_increase` на `crime_rate`? но политический климат, скорее всего, влияет и на `min_wage_increase`, и на `crime_rate`, а мы это не измеряли.

```
rows=1542 cols=3 structure=cross_section
col median_income numeric uniq=1542 range=[1.704e+04..7.861e+04]
col min_wage_increase binary uniq=2 range=[0..1] values=0|1
col crime_rate numeric uniq=1539 range=[2.4..847.2]
```

**Parse:** `<T:effect> treatment=`min_wage_increase` outcome=`crime_rate` randomized=`no` hidden=`yes``

```
size rows=1542 covariates=1
overlap auc=0.594 extreme=0 maxsmd=0.328
nonlinearity r2_linear=0.106 r2_boosting=-0.0503
flags good_overlap imbalance linear_ok
allowed sensitivity_analysis
warning ipw matching psm regression causal_forest dml doubly_robust
```

**Methods:** {'sensitivity_analysis': 1.0}

**Explanation:**

Эффект `min_wage_increase` на `crime_rate` нельзя идентифицировать по этим данным. Неизмеренный конфаундер, о котором вы пишете, влияет на оба, а эксперимента, инструмента или медиатора, чтобы разорвать эту связь, нет: регрессия, мэтчинг и DML будут смещены. В данных 1542 строк. Что можно сделать: анализ чувствительности покажет, насколько сильным должен быть скрытый конфаундер, чтобы объяснить связь; лучше провести эксперимент, найти инструмент или измерить конфаундер.

## 19. backdoor_large (en, mode=options)

**Question:** How much did `push_enabled` change time in app (`minutes_in_app`)? Note that `push_enabled` was not randomized: users chose it themselves.

```
rows=5240 cols=27 structure=cross_section
col prior_purchases integer uniq=8 range=[0..7]
col kpi_6 numeric uniq=5100 range=[12.28..193.9]
col survey_q7 numeric uniq=523 range=[0..6.36]
col device_price numeric uniq=5014 range=[0..1199]
col idx2 numeric uniq=4988 range=[1.399..104.2]
col user_age numeric uniq=2548 range=[1.18..59.27]
col idx17 numeric uniq=4983 range=[0.676..105.8]
col metric_10 numeric uniq=5102 range=[4.976..183.7]
col kpi_15 numeric uniq=3968 range=[8.32..185.7]
col country_tier categorical uniq=3 values=tier1|tier3|tier2
col kpi_13 numeric uniq=4014 range=[0..184.7]
col idx11 numeric uniq=3414 range=[0..105.7]
col account_age numeric uniq=4445 range=[0..413.8]
col sessions_7d integer uniq=18 range=[0..17]
col idx5 numeric uniq=5011 range=[0..99.68]
col survey_q14 numeric uniq=2841 range=[0..6.603]
col metric_3 numeric uniq=3948 range=[16.72..187.6]
col n_friends integer uniq=45 range=[0..45]
col idx9 numeric uniq=3387 range=[0..104.8]
col index_4 numeric uniq=3441 range=[0..102.7]
col push_enabled binary uniq=2 range=[0..1] values=1|0
col idx12 numeric uniq=5022 range=[0..116.4]
col survey_q1 numeric uniq=530 range=[0..6.62]
col minutes_in_app numeric uniq=4786 range=[0.16..82.14]
col metric_8 numeric uniq=5103 range=[11.02..186.7]
col metric_16 numeric uniq=5093 range=[14.34..187.4]
col os categorical uniq=3 values=web|ios|android
```

**Parse:** `<T:effect> treatment=`push_enabled` outcome=`minutes_in_app` randomized=`no``

```
size rows=5240 covariates=25
overlap auc=0.765 extreme=0.0147 maxsmd=0.892
nonlinearity r2_linear=0.24 r2_boosting=0.811
flags good_overlap high_dim imbalance large_sample nonlinear
allowed dml doubly_robust causal_forest psm ipw sensitivity_analysis
warning matching regression
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

Several methods fit this data about equally well: Double machine learning and Doubly robust estimation (AIPW / DR-learner).
The outcome depends on the covariates non-linearly: boosting reaches R²=0.811 vs 0.24 for a linear model.
- Double machine learning: flexible ML models capture the non-linear relationships found in the data
- Doubly robust estimation (AIPW / DR-learner): flexible models in both parts capture the non-linear relationships
Assumptions to check: all confounders are observed; the ML models predict outcome and treatment reasonably well.

## 20. backdoor_small (en, mode=recommend)

**Question:** I'm analysing our education data. What's the impact of smaller class size on graduation rate (`graduated_pct`) Thanks

```
rows=475 cols=6 structure=cross_section
col previous_grade numeric uniq=180 range=[1.35..4.55]
col school_type categorical uniq=3 values=public|private|charter
col days_absent integer uniq=15 range=[0..14]
col small_class binary uniq=2 range=[0..1] values=0|1
col graduated_pct numeric uniq=471 range=[63.33..102.3]
col sex binary uniq=2 values=female|male
```

**Parse:** `<T:effect> treatment=`small_class` outcome=`graduated_pct``

```
size rows=475 covariates=4
overlap auc=0.655 extreme=0 maxsmd=0.363
nonlinearity r2_linear=0.531 r2_boosting=0.422
flags good_overlap imbalance linear_ok
allowed matching regression sensitivity_analysis
warning ipw psm dml doubly_robust diff_in_means
```

**Methods:** {'regression': 1.0, 'psm': 0.6, 'ipw': 0.5, 'matching': 0.6, 'doubly_robust': 0.4}

**Explanation:**

Use Linear regression with an adjustment set.
Regress the outcome on the treatment plus a set of confounders that blocks all back-door paths.
A linear model fits the outcome about as well as boosting (R²=0.531 vs 0.422).
Why it fits: relationships look linear, so a linear model is adequate.
Alternative: Propensity score matching, since propensity distributions of the groups overlap.
Not recommended: Difference in means (randomized experiment), because groups differ on covariates, so randomization is questionable.
Assumptions to check: no mediators or colliders are included as controls; the relationship is approximately linear (or correctly specified).

## 21. backdoor_targeted (en, mode=recommend)

**Question:** My manager asked me this: Did `d` actually work? I want to know its impact on outcome. What's the right methodology here?

```
rows=1433 cols=10 structure=cross_section
col var1 numeric uniq=662 range=[0..11.81]
col var2 binary uniq=2 values=yes|no
col var3 numeric uniq=1360 range=[97.07..313.1]
col var4 categorical uniq=3 values=public|none|private
col var5 numeric uniq=1370 range=[11.21..41]
col var6 numeric uniq=1417 range=[3.292..101.1]
col var7 integer uniq=7 range=[0..6]
col d binary uniq=2 values=True|False
col outcome numeric uniq=1179 range=[11.56..68.84]
col var8 binary uniq=2 values=female|male
```

**Parse:** `<T:effect> treatment=`d` outcome=`outcome``

```
size rows=1433 covariates=8
overlap auc=0.84 extreme=0.083 maxsmd=0.94
nonlinearity r2_linear=0.817 r2_boosting=0.748
flags imbalance linear_ok
allowed regression doubly_robust ipw matching psm sensitivity_analysis
warning causal_forest dml diff_in_means
```

**Methods:** {'regression': 1.0, 'psm': 0.6, 'ipw': 0.5, 'matching': 0.3, 'doubly_robust': 0.4}

**Explanation:**

I recommend Linear regression with an adjustment set.
A linear model fits the outcome about as well as boosting (R²=0.817 vs 0.748).
Why it fits: relationships look linear, so a linear model is adequate.
Alternative: Propensity score matching.
Avoid: Difference in means (randomized experiment), because groups differ on covariates, so randomization is questionable.
Assumptions to check: no mediators or colliders are included as controls; the relationship is approximately linear (or correctly specified).

## 22. continuous_treatment (en, mode=recommend)

**Question:** need help: what happened to graduation rate becausse of years of education? how should i analyse this

```
rows=10176 cols=13 structure=cross_section
col class_size integer uniq=30 range=[10..41]
col parent_education_years numeric uniq=1282 range=[2.58..22.59]
col family_income numeric uniq=10160 range=[0..1.356e+05]
col age numeric uniq=4774 range=[10.06..21.01]
col previous_grade numeric uniq=316 range=[1.05..5.03]
col school_type categorical uniq=3 values=charter|public|private
col gender binary uniq=2 values=male|female
col days_absent integer uniq=18 range=[0..17]
col q1_score numeric uniq=3733 range=[0..6.656]
col index_2 numeric uniq=9241 range=[0..107.8]
col index_3 numeric uniq=9306 range=[0..106.4]
col schooling numeric uniq=1593 range=[1.03..26.4]
col graduation_rate numeric uniq=2907 range=[55.27..108]
```

**Parse:** `<T:effect> treatment=`schooling` outcome=`graduation_rate``

```
size rows=10176 covariates=11
nonlinearity r2_linear=0.929 r2_boosting=0.828
flags large_sample linear_ok
allowed causal_forest dml regression sensitivity_analysis
warning -
```

**Methods:** {'regression': 1.0, 'dml': 0.8}

**Explanation:**

Recommended method: Linear regression with an adjustment set.
A linear model fits the outcome about as well as boosting (R²=0.929 vs 0.828).
The data has 10176 rows.
Why it fits: relationships look linear, so a linear model is adequate.
Alternative: Double machine learning, since DML shines on large samples.
Assumptions to check: all confounders are observed and included; no mediators or colliders are included as controls.

## 23. discovery_gaussian (ru, mode=options)

**Question:** Какие из переменных являются причинами `Store Age`? Как это правильно проанализировать?

```
rows=784 cols=8 structure=cross_section
col City Population numeric uniq=784 range=[1.03e+04..4.663e+05]
col Store Size Sqm numeric uniq=782 range=[277.9..2285]
col Customer Age numeric uniq=722 range=[1.92..77.94]
col Avg Check numeric uniq=765 range=[6.905..59.91]
col Store Age numeric uniq=764 range=[0.12..35.61]
col Units Sold numeric uniq=779 range=[83.98..917]
col Weekly Sales numeric uniq=784 range=[2531..1.83e+04]
col Staff integer uniq=47 range=[0..54]
```

**Parse:** `<T:discovery>`

```
size rows=784 covariates=8
normality nongaussian=0
flags gaussian_data
allowed -
warning ges fci notears pc lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

Здесь подходят несколько методов примерно одинаково хорошо: Алгоритм PC и Жадный поиск по классам эквивалентности (GES).
0% непрерывных переменных заметно негауссовские.
В данных 784 строк.
- Алгоритм PC: Поиск структуры по тестам условной независимости.
- Жадный поиск по классам эквивалентности (GES): данные близки к нормальным
Проверьте допущения: нет скрытых общих причин; faithfulness: независимости в данных порождены графом.

## 24. discovery_hidden (en, mode=recommend)

**Question:** We have data on customers. Which variables cause which in this dataset? Some common causes might be missing

```
rows=751 cols=9 structure=cross_section
col Cart Value numeric uniq=706 range=[2.02..150.1]
col Sessions numeric uniq=747 range=[226.7..1465]
col Household Income numeric uniq=751 range=[9884..1.046e+05]
col Historic Spend numeric uniq=751 range=[2.4..821.9]
col Conversion Rate numeric uniq=749 range=[67.91..273.6]
col Age numeric uniq=658 range=[0.22..121.7]
col Tenure Months numeric uniq=669 range=[1.91..38.01]
col Platform categorical uniq=3 values=mobile|desktop|tablet
col Customer Id integer uniq=751 range=[1000..1750]
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=751 covariates=9
normality nongaussian=0.75
flags non_gaussian
allowed -
warning fci lingam ges notears pc
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

I recommend Fast causal inference (FCI).
75% of the continuous variables are clearly non-Gaussian.
Why it fits: designed for data with possible hidden confounders.
Works best with at least 1000 rows, has 751.
Be careful with: LiNGAM, because LiNGAM assumes no hidden confounders.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 25. discovery_nongaussian (en, mode=recommend)

**Question:** how are units sold and revenue causally connected? which method should i use?

```
rows=1271 cols=7 structure=cross_section
col Revenue numeric uniq=1271 range=[50..2.369e+04]
col Customer Age numeric uniq=1097 range=[0.24..113.2]
col Store Age Years numeric uniq=971 range=[0.12..47.63]
col Units Sold numeric uniq=1249 range=[2.4..1289]
col Avg Check numeric uniq=1238 range=[2.167..79.9]
col Population numeric uniq=1271 range=[1600..4.826e+05]
col Parking Spaces integer uniq=125 range=[0..147]
```

**Parse:** `<T:discovery>`

```
size rows=1271 covariates=7
normality nongaussian=0.857
flags non_gaussian
allowed lingam fci ges pc
warning notears
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

I recommend LiNGAM.
86% of the continuous variables are clearly non-Gaussian.
The data has 1271 rows.
Why it fits: non-Gaussian data lets LiNGAM orient every edge.
Alternative: PC algorithm.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: relationships are linear; noise terms are non-Gaussian.

## 26. front_door (en, mode=recommend)

**Question:** Nede help: Is premium trial (`trial`) worth it in terms of time in app? But tech savviness probably affects both prmium trial (`trial`) and time in app, and we didn't measure it. The whole effect of premium trial (`trial`) on time in app goes through feature usage (`feature_clicks`). Any advice?

```
rows=6615 cols=5 structure=cross_section
col os categorical uniq=3 values=ios|android|web
col trial binary uniq=2 values=yes|no
col session_minutes numeric uniq=2799 range=[0.16..64.69]
col user_id integer uniq=6615 range=[1000..7614]
col feature_clicks numeric uniq=5327 range=[0.08..29.75]
```

**Parse:** `<T:effect> treatment=`trial` outcome=`session_minutes` mediator=`feature_clicks` hidden=`yes``

```
size rows=6615 covariates=2
overlap auc=0.503 extreme=0 maxsmd=0.0398
nonlinearity r2_linear=0.296 r2_boosting=0.242
flags good_overlap large_sample linear_ok
allowed front_door sensitivity_analysis
warning causal_forest diff_in_means dml doubly_robust ipw matching psm regression
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Use Front-door adjustment.
If the whole effect flows through an observed mediator that is not confounded with the outcome, the effect is identified despite hidden treatment-outcome confounding.
A linear model fits the outcome about as well as boosting (R²=0.296 vs 0.242).
Covariates predict the treatment with AUC=0.503; 0% of units have extreme propensity scores.
The data has 6615 rows.
Why it fits: front-door works where back-door adjustment is impossible.
Avoid: Double machine learning, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: the mediator captures the entire effect of the treatment on the outcome; all back-door paths from mediator to outcome are blocked by the treatment.

## 27. instrumental (en, mode=recommend)

**Question:** I'm analysing our retail data. Should we roll out shelf price? Does it move units sold? We don't observe local demand, which likely influences who gets shelf price and also units sold. Distance to the distribution center influences shelf price and is unrelated to everything else that affects units sold. What's the right methodology here?

```
rows=1091 cols=6 structure=cross_section
col age numeric uniq=926 range=[0.62..84.69] missing=5%
col n_employees integer uniq=46 range=[0..48]
col shelf_price numeric uniq=1079 range=[0.2..106.8]
col dist_to_dc_km numeric uniq=1079 range=[0..97.54]
col units_sold numeric uniq=1070 range=[143.3..884.8]
col parking_spaces integer uniq=122 range=[0..166]
```

**Parse:** `<T:effect> treatment=`shelf_price` outcome=`units_sold` instrument=`dist_to_dc_km` hidden=`yes``

```
size rows=1091 covariates=3
firststage F=158.0
nonlinearity r2_linear=0.782 r2_boosting=0.729
flags linear_ok
allowed iv sensitivity_analysis
warning regression causal_forest dml
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Best fit: Instrumental variables (2SLS).
Use a variable that shifts the treatment but affects the outcome only through it, so the effect is identified even with unobserved confounders.
The instrument is strong: first-stage F=158.0.
A linear model fits the outcome about as well as boosting (R²=0.782 vs 0.729).
Why it fits: IV is one of the few ways to handle unobserved confounders.
Avoid: Linear regression with an adjustment set, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: relevance (the instrument really moves the treatment); exclusion restriction (the instrument affects the outcome only through the treatment).

## 28. not_causal (en, mode=refuse)

**Question:** We have data on customers. Can you build a model to predict basket size? Which method should I use

```
rows=118 cols=8 structure=cross_section
col Avg Check numeric uniq=116 range=[9.8..54.81]
col Population numeric uniq=118 range=[4.374e+04..4.551e+05]
col Area categorical uniq=4 values=north|west|east|south
col N Employees integer uniq=31 range=[8..42]
col Loyalty Member binary uniq=2 range=[0..1] values=0|1
col Parking Spaces integer uniq=64 range=[7..129]
col N Competitors integer uniq=10 range=[0..9]
col Customer Age numeric uniq=117 range=[13.7..67.34]
```

**Parse:** `<T:not_causal> target=`Avg Check``

```
size rows=118 covariates=8
flags small_sample
allowed -
warning -
```

**Methods:** {}

**Explanation:**

This is a prediction question, not a causal one: it asks how well `Avg Check` can be predicted or described, not what happens if you change something. Use a predictive model (for example gradient boosting or regularised regression) with cross-validation, or descriptive statistics. Causal methods are needed only if you plan to intervene on a variable and want the effect of that change.

## 29. panel_factor (en, mode=options)

**Question:** I'm analysing our health data. Smoking cessation program was introduced in some hospitals. What did it do to `readmissionRate`?

```
rows=819 cols=5 structure=panel
panel unit=hospitalId(39) time=year(21) balanced
col hospitalId categorical uniq=39 values=H001|H002|H003|H004|H005
col readmissionRate numeric uniq=753 range=[0.2..99.09]
col cessationProgram binary uniq=2 values=False|True
col year integer uniq=21 range=[1988..2008]
col after binary uniq=2 values=yes|no
design group_post(cessationProgram,after) treated=3 controls=36 pre=10 post=11 block
```

**Parse:** `<T:effect> treatment=`cessationProgram` outcome=`readmissionRate``

```
size rows=819 covariates=0 treated=3 controls=36 pre=10 post=11
pretrend drift=-20.5 rel=1.24 p=1.44e-08
scfit r2=1 donors=8 outside=0
flags has_controls policy_design pretrend_violated sc_good_fit
allowed synthetic_control sdid sensitivity_analysis
warning did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.9, 'did': 0.1}

**Explanation:**

There is no single best choice here; Synthetic difference-in-differences and Synthetic control are both good options.
Before the treatment the gap between treated and control units drifted by -20.5 (124% of the outcome's SD, p=1.44e-08), so the trends are not parallel.
A synthetic control reproduces the treated path before the treatment well (R²=1).
There are 3 treated units and 36 control units, 10 periods before and 11 after the treatment.
- Synthetic difference-in-differences: unit and time weights repair non-parallel trends
- Synthetic control: does not require parallel trends, it matches the trajectory instead
Why Synthetic difference-in-differences may be better than Synthetic control: lower variance thanks to fixed effects, and levels need not match.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 30. panel_parallel (en, mode=recommend)

**Question:** Does `loyaltyMember` have a positive effect on units sold? Any advice?

```
rows=2236 cols=6 structure=panel
panel unit=store(86) time=yr(26) balanced
col store categorical uniq=86 values=store_1|store_2|store_3|store_4|store_5
col yr integer uniq=26 range=[1995..2020]
col quantity numeric uniq=2193 range=[2.4..998.2]
col loyaltyMember binary uniq=2 range=[0..1] values=0|1
col population numeric uniq=2125 range=[1799..5.294e+05] missing=5%
col customerAge numeric uniq=2033 range=[0..85.39] missing=6%
design indicator(loyaltyMember) treated=36 controls=50 pre=17 post=9 block
```

**Parse:** `<T:effect> treatment=`loyaltyMember` outcome=`quantity``

```
size rows=2236 covariates=2 treated=36 controls=50 pre=17 post=9
pretrend drift=-4.92 rel=0.0289 p=0.595
scfit r2=-11.6 donors=50 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_poor_fit
allowed did sdid sensitivity_analysis
warning dml its regression synthetic_control
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

I recommend Difference-in-differences.
Compare the change in outcome before and after treatment in the treated group with the same change in the control group.
Pre-treatment trends look parallel (drift -4.92, p=0.595).
Why it fits: pre-treatment trends of the groups are parallel; many treated units make the estimate precise.
Alternative: Synthetic difference-in-differences.
Not recommended: Synthetic control, because the synthetic control does not reproduce the pre-treatment path.
Assumptions to check: parallel trends (without treatment, both groups would have moved in parallel); no anticipation of the treatment.

## 31. panel_staggered (en, mode=recommend)

**Question:** My manager asked me this: Compare countries with and wiithout `pushEnabled`: what's the efffect on `subscriptions`? Thanks!

```
rows=384 cols=4 structure=panel
panel unit=market(32) time=month(12) balanced
col subscriptions numeric uniq=372 range=[3.04..113.8]
col pushEnabled binary uniq=2 values=False|True
col market integer uniq=32 range=[1..32]
col month datetime uniq=12 values=2017-08-01|2018-07-01
design indicator(pushEnabled) treated=25 controls=7 pre=3 post=9 staggered
```

**Parse:** `<T:effect> treatment=`pushEnabled` outcome=`subscriptions``

```
size rows=384 covariates=0 treated=25 controls=7 pre=3 post=9
pretrend drift=-0.957 rel=0.0627 p=0.85
scfit r2=1 donors=6 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit short_pre_period staggered
allowed sensitivity_analysis
warning did sdid dml regression synthetic_control
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Use Synthetic difference-in-differences.
Weighted DiD with unit weights (like synthetic control, but only trends must match, not levels) and time weights that focus on pre-periods most similar to the post-period.
Pre-treatment trends look parallel (drift -0.957, p=0.85).
A synthetic control reproduces the treated path before the treatment well (R²=1).
There are 25 treated units and 7 control units, 3 periods before and 9 after the treatment.
Why it fits: staggered adoption is handled by running SDID per adoption cohort and averaging.
Works best with at least 5 pre-treatment periods, has 3.
Not recommended: Difference-in-differences, because naive two-way fixed effects is biased when units are treated at different times.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 32. rct (en, mode=recommend)

**Question:** Estimate how new onboarding flow influenced subscriptions. New onboarding flow was assigned at random. Any advice

```
rows=3040 cols=9 structure=cross_section
col prior_purchases integer uniq=8 range=[0..7]
col n_friends integer uniq=42 range=[0..45]
col os categorical uniq=3 values=ios|web|android
col paid_conversions numeric uniq=2955 range=[6.992..112.7]
col variant_b binary uniq=2 values=control|treated
col days_since_signup numeric uniq=2718 range=[0..387.3]
col device_price numeric uniq=3017 range=[0..1126]
col prev_sessions integer uniq=17 range=[0..16]
col age numeric uniq=1943 range=[0.95..57.49]
```

**Parse:** `<T:effect> treatment=`variant_b` outcome=`paid_conversions` randomized=`yes``

```
size rows=3040 covariates=7
overlap auc=0.501 extreme=0 maxsmd=0.08
nonlinearity r2_linear=0.693 r2_boosting=0.632
flags good_overlap linear_ok randomization_plausible
allowed regression diff_in_means ipw matching psm causal_forest dml doubly_robust sensitivity_analysis
warning -
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

I recommend Difference in means (randomized experiment).
A linear model fits the outcome about as well as boosting (R²=0.693 vs 0.632).
Why it fits: covariates are balanced, so randomization looks credible.
Alternative: Linear regression with an adjustment set, since relationships look linear, so a linear model is adequate.
Assumptions to check: SUTVA (one unit's treatment does not affect another unit's outcome); no differential attrition between groups.

## 33. rct_heterogeneous (ru, mode=options)

**Question:** Привет! Хочу оценить индивидуальный эффект `telehealth` на `readmissions`.

```
rows=4525 cols=10 structure=cross_section
col telehealth binary uniq=2 values=no|yes
col readmissions numeric uniq=4216 range=[2.108..72.7]
col cholesterol numeric uniq=3780 range=[73.41..315.1]
col gender binary uniq=2 values=female|male
col smoker binary uniq=2 values=yes|no
col comorbidities integer uniq=7 range=[0..6]
col insurance categorical uniq=3 values=public|private|none
col baseline_score numeric uniq=872 range=[0..11.95] missing=6%
col bmi numeric uniq=3925 range=[7.718..40.7]
col возраст numeric uniq=3025 range=[12.19..106]
```

**Parse:** `<T:heterogeneity> treatment=`telehealth` outcome=`readmissions``

```
size rows=4525 covariates=8
overlap auc=0.516 extreme=0 maxsmd=0.0911
nonlinearity r2_linear=0.784 r2_boosting=0.773
flags good_overlap linear_ok randomization_plausible
allowed meta_learners causal_forest dml doubly_robust
warning -
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

Здесь подходят несколько методов примерно одинаково хорошо: Мета-лёрнеры для неоднородного эффекта и Причинный лес.
Линейная модель описывает исход почти так же хорошо, как бустинг (R²=0.784 против 0.773).
Ковариаты предсказывают воздействие с AUC=0.516; у 0% объектов экстремальный propensity score.
В данных 4525 строк.
- Мета-лёрнеры для неоднородного эффекта: ковариаты сбалансированы, рандомизация выглядит правдоподобно
- Причинный лес: Ансамбль причинных деревьев, который находит подгруппы с разным эффектом.
Проверьте допущения: нет ненаблюдаемых конфаундеров (в эксперименте выполняется автоматически); группы пересекаются по ковариатам.

## 34. rdd (en, mode=recommend)

**Question:** Did medicaid expansion increase GDP per capita? Eligibility for medicaid expansion was `poverty_rate` >= 16.1. Any advice?

```
rows=18819 cols=4 structure=cross_section
col unemployment numeric uniq=7336 range=[0..15.28]
col poverty_rate numeric uniq=10832 range=[0.08..35.15]
col medicaid_expansion binary uniq=2 values=yes|no
col gdp_per_capita numeric uniq=18816 range=[1.971e+04..8.164e+04]
```

**Parse:** `<T:effect> treatment=`medicaid_expansion` outcome=`gdp_per_capita` running=`poverty_rate``

```
size rows=18819 covariates=2
overlap auc=1 extreme=0.899 maxsmd=2.68
threshold var=poverty_rate cutoff=16.0 acc=0.996 jump=0.978
nonlinearity r2_linear=0.709 r2_boosting=0.651
flags imbalance large_sample linear_ok poor_overlap running_variable_found sharp_cutoff
allowed rdd sensitivity_analysis
warning causal_forest dml doubly_robust regression diff_in_means ipw matching psm
```

**Methods:** {'rdd': 1.0}

**Explanation:**

I recommend Regression discontinuity design.
When treatment is assigned by a threshold on a running variable, units just above and just below the threshold are comparable.
Treatment switches at `poverty_rate` >= 16.0: this rule explains 100% of assignments.
Why it fits: treatment jumps at a threshold of a continuous variable; the threshold determines treatment almost perfectly (sharp RDD).
Not recommended: Doubly robust estimation (AIPW / DR-learner), because the weighting part still suffers from extreme weights.
Assumptions to check: other factors change smoothly at the cutoff; the effect is estimated only for units near the cutoff.

## 35. time_series (en, mode=recommend)

**Question:** I have a CSV with 347 rows. What's the best way to measure the effect of the premium trial on subscriptions (Paid Conversions) with this data? Which method should I use

```
rows=347 cols=4 structure=time_series
time_series time=date
col date datetime uniq=347 values=2020-05-17|2021-04-28
col Premium Trial binary uniq=2 values=False|True
col Paid Conversions numeric uniq=344 range=[53.73..89.31]
col Prior Purchases numeric uniq=205 range=[0.03..5.04]
```

**Parse:** `<T:effect> treatment=`Premium Trial` outcome=`Paid Conversions``

```
size rows=347 covariates=1 pre=215 post=132
flags no_controls
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Recommended method: Interrupted time series / CausalImpact.
This is a single series with no control group: 215 periods before the intervention and 132 after.
Why it fits: there is no control group, so a time-series counterfactual is the main option.
Assumptions to check: the pre-treatment model extrapolates correctly into the post-period; no other event coincides with the intervention.

## 36. unidentifiable (en, mode=refuse)

**Question:** We have data on counties. Does the minimum wage increase have a positive effect on crime_rate? I worry that unmeasured political climate drives both the minimum wage increase and crime_rate. Note that the minimum wage increase was not randomized: counties chose it themselves.

```
rows=1542 cols=3 structure=cross_section
col median_income numeric uniq=1542 range=[1.704e+04..7.861e+04]
col min_wage_increase binary uniq=2 range=[0..1] values=0|1
col crime_rate numeric uniq=1539 range=[2.4..847.2]
```

**Parse:** `<T:effect> treatment=`min_wage_increase` outcome=`crime_rate` randomized=`no` hidden=`yes``

```
size rows=1542 covariates=1
overlap auc=0.594 extreme=0 maxsmd=0.328
nonlinearity r2_linear=0.106 r2_boosting=-0.0503
flags good_overlap imbalance linear_ok
allowed sensitivity_analysis
warning ipw matching psm regression causal_forest dml doubly_robust
```

**Methods:** {'sensitivity_analysis': 1.0}

**Explanation:**

The effect of `min_wage_increase` on `crime_rate` cannot be identified from this data. The unmeasured confounder you mention affects both, and there is no experiment, instrument or mediator to break that link, so regression, matching or DML would all be biased. The data has 1542 rows. What you can do: run a sensitivity analysis to see how strong the hidden confounder would have to be to explain the association away; better, run an experiment, find a valid instrument, or measure the confounder.

## 37. backdoor_large (en, mode=options)

**Question:** We have data on customers. Does redesigned affect revenue? Note that redesigned was not randomized: customers chose it themselves. Which method should I use?

```
rows=31432 cols=17 structure=cross_section
col store_size_sqm numeric uniq=30967 range=[0..2359]
col city_population numeric uniq=31387 range=[0..6.214e+05]
col store_age numeric uniq=2780 range=[0..37.09]
col n_competitors integer uniq=14 range=[0..14]
col customer_age numeric uniq=5646 range=[0..90.58]
col region categorical uniq=4 values=east|west|north|south
col n_employees integer uniq=57 range=[0..57]
col prior_orders integer uniq=24 range=[0..23]
col parking_spaces integer uniq=154 range=[0..172]
col metric_1 numeric uniq=9955 range=[0..216.2]
col survey_q2 numeric uniq=622 range=[0..6.89]
col metric_3 numeric uniq=9968 range=[0..196.3]
col metric_4 numeric uniq=26689 range=[0..203.5]
col q5_score numeric uniq=620 range=[0..6.94]
col q6_score numeric uniq=4839 range=[0..7.064]
col redesigned binary uniq=2 values=True|False
col revenue numeric uniq=31385 range=[50..2.757e+04]
```

**Parse:** `<T:effect> treatment=`redesigned` outcome=`revenue` randomized=`no``

```
size rows=31432 covariates=15
overlap auc=0.641 extreme=0 maxsmd=0.372
nonlinearity r2_linear=0.294 r2_boosting=0.794
flags good_overlap imbalance large_sample nonlinear
allowed causal_forest dml doubly_robust ipw matching psm sensitivity_analysis
warning regression
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

Several methods fit this data about equally well: Double machine learning and Doubly robust estimation (AIPW / DR-learner).
The outcome depends on the covariates non-linearly: boosting reaches R²=0.794 vs 0.294 for a linear model.
- Double machine learning: flexible ML models capture the non-linear relationships found in the data
- Doubly robust estimation (AIPW / DR-learner): flexible models in both parts capture the non-linear relationships
Assumptions to check: the ML models predict outcome and treatment reasonably well; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 38. backdoor_small (en, mode=options)

**Question:** We have data on employees. Did remote_work increase employee turnover? Which method should I use?

```
rows=141 cols=6 structure=cross_section
col education categorical uniq=4 values=school|bachelor|phd|master
col attrition numeric uniq=136 range=[1.67..29.98]
col dept categorical uniq=4 values=sales|finance|ops|it
col company_size integer uniq=122 range=[0..1159]
col age numeric uniq=140 range=[17.24..58.46]
col remote_work binary uniq=2 range=[0..1] values=0|1
```

**Parse:** `<T:effect> treatment=`remote_work` outcome=`attrition` randomized=`no``

```
size rows=141 covariates=4
overlap auc=0.848 extreme=0.113 maxsmd=0.841
nonlinearity r2_linear=0.755 r2_boosting=0.646
flags imbalance linear_ok poor_overlap small_sample
allowed sensitivity_analysis
warning regression ipw matching psm
```

**Methods:** {'psm': 1.0, 'matching': 0.9, 'regression': 0.5, 'ipw': 0.2}

**Explanation:**

The treatment here is almost decided by the covariates, so the treated and untreated groups barely overlap.
Covariates predict the treatment with AUC=0.848; 11% of units have extreme propensity scores. For these units there is no comparable unit in the other group, so the effect for everyone can only be extrapolated.
Options:
- Propensity score matching with a caliper: units without a close match are dropped, so the effect is for the matched units, usually the treated.
- Covariate matching with a caliper: units without a comparable match are dropped, so the effect is for the matched units only.
Be careful with: Inverse probability weighting, because extreme weights make the estimate unstable.
Whichever you choose, report how many units are dropped and how they differ from the rest.
Assumptions to check: all confounders are observed; the propensity score model is correctly specified.

## 39. backdoor_targeted (en, mode=options)

**Question:** We have data on students. What's the causal effect of `treatment` on `outcome`?

```
rows=15856 cols=13 structure=cross_section
col outcome numeric uniq=13349 range=[22.77..112.9]
col f7 numeric uniq=5720 range=[10.25..21.45]
col f8 integer uniq=31 range=[10..40]
col f9 numeric uniq=5729 range=[0..108.8]
col obs integer uniq=15856 range=[1000..1.686e+04]
col f10 numeric uniq=14532 range=[2.266..198]
col treatment binary uniq=2 values=no|yes
col f3 binary uniq=2 values=female|male
col f1 categorical uniq=3 values=public|charter|private
col f6 numeric uniq=7805 range=[4.357..23.59]
col f5 integer uniq=19 range=[0..19]
col f2 numeric uniq=15830 range=[0..1.291e+05]
col f4 numeric uniq=325 range=[1.1..4.73]
```

**Parse:** `<T:effect> treatment=`treatment` outcome=`outcome``

```
size rows=15856 covariates=11
overlap auc=0.979 extreme=0.666 maxsmd=1.34
nonlinearity r2_linear=0.947 r2_boosting=0.903
flags imbalance large_sample linear_ok poor_overlap
allowed sensitivity_analysis
warning causal_forest dml doubly_robust regression diff_in_means ipw matching psm
```

**Methods:** {'doubly_robust': 1.0, 'psm': 0.9, 'matching': 0.3, 'regression': 0.5, 'ipw': 0.2}

**Explanation:**

The groups barely overlap, so there is no single safe answer: the right method depends on which effect you need.
Covariates predict the treatment with AUC=0.979; 67% of units have extreme propensity scores. For these units there is no comparable unit in the other group, so the effect for everyone can only be extrapolated.
Options:
- Doubly robust estimation (AIPW / DR-learner) after trimming the units with extreme propensity scores: the effect for units that could have gone either way, not for everyone.
- Propensity score matching with a caliper: units without a close match are dropped, so the effect is for the matched units, usually the treated.
Not recommended: Inverse probability weighting, because extreme weights make the estimate unstable.
Whichever you choose, report how many units are dropped and how they differ from the rest.
Assumptions to check: overlap between groups; at least one of the two models is correct.

## 40. continuous_treatment (en, mode=recommend)

**Question:** My manager asked me this: Impressions -> site visits: is there an effect? What's the right methodology here?

```
rows=19188 cols=14 structure=cross_section
col sessions numeric uniq=18933 range=[282..1949]
col subscribed binary uniq=2 values=yes|no
col household_income numeric uniq=19136 range=[0..1.236e+05]
col impressions numeric uniq=19127 range=[120..6.484e+04]
col q1_score numeric uniq=600 range=[0..6.74]
col past_orders integer uniq=19 range=[0..18]
col idx2 numeric uniq=16110 range=[0..107.7]
col age numeric uniq=15225 range=[0..78.79]
col cart_value numeric uniq=7310 range=[0..137.8]
col prev_spend numeric uniq=15359 range=[0..745.5]
col platform categorical uniq=3 values=desktop|mobile|tablet
col tenure_months numeric uniq=4049 range=[0..63.63]
col acquisition_channel categorical uniq=4 values=social|direct|search|referral
col kpi_3 numeric uniq=17311 range=[10.51..192.8]
```

**Parse:** `<T:effect> treatment=`impressions` outcome=`sessions``

```
size rows=19188 covariates=12
nonlinearity r2_linear=0.735 r2_boosting=0.896
flags large_sample nonlinear
allowed causal_forest dml sensitivity_analysis
warning regression
```

**Methods:** {'dml': 1.0, 'causal_forest': 0.6, 'regression': 0.3}

**Explanation:**

Recommended method: Double machine learning.
The outcome depends on the covariates non-linearly: boosting reaches R²=0.896 vs 0.735 for a linear model.
The data has 19188 rows.
Why it fits: flexible ML models capture the non-linear relationships found in the data; DML shines on large samples.
Alternative: Causal forest (CausalForestDML), since trees capture non-linear relationships automatically.
Not recommended: Linear regression with an adjustment set, because the outcome depends on the covariates non-linearly, so a linear model is misspecified.
Assumptions to check: all confounders are observed; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 41. discovery_gaussian (en, mode=options)

**Question:** I have a CSV with 3390 rows. I don't have a hypothesis yet. What causes what among user age (`User Age`), days since signup (`Account Age`) and Subscriptions

```
rows=3390 cols=9 structure=cross_section
col Idx1 numeric uniq=2510 range=[0.3..99.21]
col Account Age numeric uniq=3168 range=[1.6..477.6]
col Subscriptions numeric uniq=2515 range=[5.01..109.4]
col User Age numeric uniq=2060 range=[1.05..54.07]
col Retention numeric uniq=2231 range=[6.01..76.56]
col Device Price numeric uniq=3382 range=[4..1551]
col Minutes In App numeric uniq=3189 range=[0.16..57.29]
col Country Tier categorical uniq=3 values=tier2|tier3|tier1
col Prior Purchases integer uniq=8 range=[0..7]
```

**Parse:** `<T:discovery>`

```
size rows=3390 covariates=9
normality nongaussian=0
flags gaussian_data
allowed ges fci notears pc
warning lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

Several methods fit this data about equally well: PC algorithm and Greedy equivalence search.
0% of the continuous variables are clearly non-Gaussian.
The data has 3390 rows.
- PC algorithm: Constraint-based discovery that removes edges using conditional independence tests and orients edges through v-structures.
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: causal sufficiency (no hidden common causes); faithfulness (independences in data come only from the graph).

## 42. discovery_hidden (en, mode=recommend)

**Question:** Discover the causal structure of recovery time, `Cholesterol`, patient age and body mass index. There may be unmeasured variables that affect several of these.

```
rows=9015 cols=8 structure=cross_section
col Recovery Days numeric uniq=6650 range=[0.08..34.29]
col Age numeric uniq=8246 range=[0.28..130.5]
col Baseline Score numeric uniq=1070 range=[0.04..16.69]
col Cholesterol numeric uniq=8691 range=[42.27..356.1]
col Index 1 numeric uniq=4563 range=[0.3..143.8]
col Readmission Rate numeric uniq=3574 range=[1.25..75.76]
col Blood Pressure numeric uniq=4355 range=[103.7..156.1]
col Bmi numeric uniq=6873 range=[19.18..34.88]
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=9015 covariates=8
normality nongaussian=1
flags large_sample non_gaussian
allowed fci
warning lingam notears pc ges
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

I recommend Fast causal inference (FCI).
An extension of PC that allows hidden confounders and marks edges that may be explained by them.
100% of the continuous variables are clearly non-Gaussian.
The data has 9015 rows.
Why it fits: designed for data with possible hidden confounders.
Be careful with: Greedy equivalence search, because GES assumes no hidden confounders.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 43. discovery_nongaussian (en, mode=recommend)

**Question:** i'm analysing our tech data. can you find the causal graph behind this data? any advice?

```
rows=1273 cols=10 structure=cross_section
col User Age numeric uniq=1036 range=[14.03..61.87]
col Retained numeric uniq=1233 range=[30.59..99.77]
col Subscriptions numeric uniq=1117 range=[10.97..119.4]
col Index 2 numeric uniq=1258 range=[7.262..92.78]
col Survey Q1 numeric uniq=1071 range=[0.02..6.69]
col Days Since Signup numeric uniq=1270 range=[1.6..538.7]
col Minutes In App numeric uniq=1042 range=[0.16..56.41]
col Device Price numeric uniq=1273 range=[4..1369]
col Friends integer uniq=41 range=[0..44]
col Country Tier categorical uniq=3 values=tier1|tier2|tier3
```

**Parse:** `<T:discovery>`

```
size rows=1273 covariates=10
normality nongaussian=0.889
flags non_gaussian
allowed lingam fci ges pc
warning notears
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

I recommend LiNGAM.
89% of the continuous variables are clearly non-Gaussian.
The data has 1273 rows.
Why it fits: non-Gaussian data lets LiNGAM orient every edge.
Alternative: PC algorithm.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: noise terms are non-Gaussian; no hidden confounders.

## 44. front_door (en, mode=recommend)

**Question:** Hello, Can we say that `promo` improves basket size? `promo` changes foot traffic, and foot traffic is what changes basket size; nothing else connects them. We don't observe manager quality, which likely influences who gets `promo` and also basket size. What's the right methodology here?

```
rows=1295 cols=5 structure=cross_section
col promo binary uniq=2 values=no|yes
col foot_traffic numeric uniq=1286 range=[829.8..5647]
col basket_size numeric uniq=1051 range=[10.43..58.3]
col parking_spaces integer uniq=122 range=[0..152] missing=6%
col past_purchases integer uniq=22 range=[0..24] missing=4%
```

**Parse:** `<T:effect> treatment=`promo` outcome=`basket_size` mediator=`foot_traffic` hidden=`yes``

```
size rows=1295 covariates=2
overlap auc=0.515 extreme=0 maxsmd=0.125
nonlinearity r2_linear=0.353 r2_boosting=0.238
flags good_overlap linear_ok
allowed front_door sensitivity_analysis
warning diff_in_means ipw matching psm regression causal_forest dml doubly_robust
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Best fit: Front-door adjustment.
A linear model fits the outcome about as well as boosting (R²=0.353 vs 0.238).
Covariates predict the treatment with AUC=0.515; 0% of units have extreme propensity scores.
Why it fits: front-door works where back-door adjustment is impossible.
Not recommended: Covariate matching, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: no unobserved confounding between treatment and mediator; all back-door paths from mediator to outcome are blocked by the treatment.

## 45. instrumental (en, mode=recommend)

**Question:** Hi! Does `tax_rate` reduce cigarette sales per capita (cigsale)? We don't observe local economic conditions, which likely influences who gets `tax_rate` and also cigarette sales per capita (cigsale). Neighbouring state's policy (`neighbor_policy`) shifts `tax_rate` but shouldn't affect cigarette sales per capita (cigsale) directly. Thanks!

```
rows=23422 cols=8 structure=cross_section
col county_id integer uniq=23422 range=[1000..2.442e+04]
col age15to24 numeric uniq=18 range=[0.08..0.26]
col avg_price numeric uniq=21041 range=[0..211.5]
col lnincome numeric uniq=23346 range=[8821..8.614e+04]
col tax_rate numeric uniq=2622 range=[0.06..40.73]
col urban_share numeric uniq=752 range=[0.255..1.163]
col neighbor_policy binary uniq=2 range=[0..1] values=0|1
col cigsale numeric uniq=21025 range=[0.6..241.3]
```

**Parse:** `<T:effect> treatment=`tax_rate` outcome=`cigsale` instrument=`neighbor_policy` hidden=`yes``

```
size rows=23422 covariates=5
firststage F=4.78e+03
nonlinearity r2_linear=0.648 r2_boosting=0.584
flags large_sample linear_ok
allowed iv sensitivity_analysis
warning causal_forest dml regression
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Recommended method: Instrumental variables (2SLS).
Use a variable that shifts the treatment but affects the outcome only through it, so the effect is identified even with unobserved confounders.
The instrument is strong: first-stage F=4.78e+03.
Why it fits: IV is one of the few ways to handle unobserved confounders.
Be careful with: Double machine learning, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: exclusion restriction (the instrument affects the outcome only through the treatment); monotonicity, for a LATE interpretation.

## 46. not_causal (en, mode=refuse)

**Question:** Hey, How accurately can we predict turnover_rate from the other columns? What's the right methodology here?

```
rows=139 cols=10 structure=cross_section
col commute_minutes numeric uniq=139 range=[0..76.62]
col experience numeric uniq=128 range=[0..19.58]
col job_training binary uniq=2 range=[0..1] values=0|1
col age numeric uniq=139 range=[8.82..60.48]
col company_size integer uniq=127 range=[0..1093]
col turnover_rate numeric uniq=133 range=[1.19..25.29]
col dept categorical uniq=4 values=sales|it|finance|ops
col team_size integer uniq=17 range=[0..18]
col education categorical uniq=4 values=phd|school|bachelor|master
col last_rating numeric uniq=136 range=[1.155..5.366]
```

**Parse:** `<T:not_causal> target=`turnover_rate``

```
size rows=139 covariates=10
flags small_sample
allowed -
warning -
```

**Methods:** {}

**Explanation:**

This is a prediction question, not a causal one: it asks how well `turnover_rate` can be predicted or described, not what happens if you change something. Use a predictive model (for example gradient boosting or regularised regression) with cross-validation, or descriptive statistics. Causal methods are needed only if you plan to intervene on a variable and want the effect of that change.

## 47. panel_factor (ru, mode=recommend)

**Question:** В части магазинов ввели `newLayout`, в остальных нет. Изменился ли `avgCheck`? Какой метод выбрать?

```
rows=1066 cols=6 structure=panel
panel unit=storeId(41) time=year(26) balanced
col newLayout binary uniq=2 values=no|yes
col year integer uniq=26 range=[1979..2004]
col staff integer uniq=49 range=[0..54]
col avgCheck numeric uniq=959 range=[0.16..108.2]
col after_treatment binary uniq=2 values=no|yes
col storeId integer uniq=41 range=[1..41]
design group_post(newLayout,after_treatment) treated=1 controls=40 pre=22 post=4 block
```

**Parse:** `<T:effect> treatment=`newLayout` outcome=`avgCheck``

```
size rows=1066 covariates=1 treated=1 controls=40 pre=22 post=4
pretrend drift=43.0 rel=3.45 p=7.71e-15
scfit r2=0.503 donors=3 outside=0.5
flags has_controls policy_design pretrend_violated sc_poor_fit single_treated_unit
allowed sdid sensitivity_analysis
warning synthetic_control did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.4, 'did': 0.1}

**Explanation:**

Рекомендуемый метод: Синтетическая разность разностей (SDID).
До воздействия разрыв между группами сместился на 43.0 (345% от SD исхода, p=7.71e-15), тренды не параллельны.
Синтетический контроль плохо повторяет траекторию до воздействия (R²=0.503).
Почему подходит: тренды до воздействия не параллельны; воздействие получил всего один объект.
Синтетический контроль здесь хуже: синтетический контроль плохо повторяет траекторию до воздействия.
Проверьте допущения: параллельные тренды после перевзвешивания объектов и периодов; нет перетока эффекта на контрольные объекты.

## 48. panel_parallel (en, mode=recommend)

**Question:** Would `avgCheck` have been different without new store layout (layoutChange)? What approach would you recommend

```
rows=492 cols=6 structure=panel
panel unit=store(82) time=week_start(6) balanced
col pastPurchases integer uniq=22 range=[0..21] missing=5%
col week_start datetime uniq=6 values=2022-08-01|2022-09-05
col layoutChange binary uniq=2 range=[0..1] values=0|1
col nEmployees integer uniq=43 range=[2..51] missing=7%
col store categorical uniq=82 values=S001|S002|S003|S004|S005
col avgCheck numeric uniq=459 range=[1.6..57.63]
design indicator(layoutChange) treated=28 controls=54 pre=4 post=2 block
```

**Parse:** `<T:effect> treatment=`layoutChange` outcome=`avgCheck``

```
size rows=492 covariates=2 treated=28 controls=54 pre=4 post=2
pretrend drift=-0.615 rel=0.0687 p=0.596
scfit r2=1 donors=8 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit short_pre_period
allowed did sensitivity_analysis
warning dml regression sdid synthetic_control
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

Recommended method: Difference-in-differences.
Compare the change in outcome before and after treatment in the treated group with the same change in the control group.
Pre-treatment trends look parallel (drift -0.615, p=0.596).
A synthetic control reproduces the treated path before the treatment well (R²=1).
Why it fits: pre-treatment trends of the groups are parallel; many treated units make the estimate precise.
Alternative: Synthetic difference-in-differences.
Avoid: Synthetic control, because too few pre-treatment periods to trust the fit.
Assumptions to check: parallel trends (without treatment, both groups would have moved in parallel); no anticipation of the treatment.

## 49. panel_staggered (en, mode=recommend)

**Question:** Hello, We track `avgCheck` for many stores over time. How did new store layout (`layoutChange`) affect it? Which method should I use?

```
rows=1012 cols=5 structure=panel
panel unit=storeId(92) time=year(11) balanced
col layoutChange binary uniq=2 values=False|True
col storeAge numeric uniq=788 range=[0..30.11]
col year integer uniq=11 range=[1994..2004]
col avgCheck numeric uniq=895 range=[14.48..76.92]
col storeId categorical uniq=92 values=store_1|store_2|store_3|store_4|store_5
design indicator(layoutChange) treated=68 controls=24 pre=5 post=6 staggered
```

**Parse:** `<T:effect> treatment=`layoutChange` outcome=`avgCheck``

```
size rows=1012 covariates=1 treated=68 controls=24 pre=5 post=6
pretrend drift=0.589 rel=0.0669 p=0.597
scfit r2=1 donors=24 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit staggered
allowed sdid sensitivity_analysis
warning did synthetic_control dml regression
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Use Synthetic difference-in-differences.
Weighted DiD with unit weights (like synthetic control, but only trends must match, not levels) and time weights that focus on pre-periods most similar to the post-period.
Pre-treatment trends look parallel (drift 0.589, p=0.597).
Why it fits: staggered adoption is handled by running SDID per adoption cohort and averaging.
Not recommended: Difference-in-differences, because naive two-way fixed effects is biased when units are treated at different times.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 50. rct (en, mode=recommend)

**Question:** My manager asked me this: Is there a caausal impact of medicaid expansion on `crime_rate`? Thanks!

```
rows=19895 cols=7 structure=cross_section
col population numeric uniq=19773 range=[0..1.275e+07]
col crime_rate numeric uniq=19412 range=[2.4..931.2]
col medicaid_expansion binary uniq=2 values=False|True
col retprice numeric uniq=9519 range=[0..222.8]
col beer numeric uniq=11114 range=[8.509..38.25]
col lnincome numeric uniq=19888 range=[1.211e+04..8.302e+04]
col young_share numeric uniq=18 range=[0.09..0.26]
```

**Parse:** `<T:effect> treatment=`medicaid_expansion` outcome=`crime_rate``

```
size rows=19895 covariates=5
overlap auc=0.505 extreme=0 maxsmd=0.0321
nonlinearity r2_linear=0.377 r2_boosting=0.328
flags good_overlap large_sample linear_ok randomization_plausible
allowed regression causal_forest dml doubly_robust ipw matching psm sensitivity_analysis
warning diff_in_means
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

Best fit: Difference in means (randomized experiment).
A linear model fits the outcome about as well as boosting (R²=0.377 vs 0.328).
Why it fits: covariates are balanced, so randomization looks credible.
Valid only if treatment was randomized; please confirm.
Alternative: Linear regression with an adjustment set, since relationships look linear, so a linear model is adequate.
Assumptions to check: SUTVA (one unit's treatment does not affect another unit's outcome); no differential attrition between groups.
