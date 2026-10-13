# SFT review sample

## 1. backdoor_large (en, mode=options)

**Question:** Which method should I use to estimate the effect of `d` on `Y`? Users opted into `d` on their own. Thanks!

```
rows=32391 cols=14 structure=cross_section
col feature_6 integer uniq=10 range=[0..9]
col feature_2 integer uniq=48 range=[0..48]
col feature_5 categorical uniq=3 values=web|ios|android
col feature_9 numeric uniq=6821 range=[0..112.9]
col feature_4 categorical uniq=3 values=tier2|tier1|tier3
col Y numeric uniq=4052 range=[0.16..96.92]
col feature_3 numeric uniq=25893 range=[0..1404]
col feature_7 numeric uniq=28470 range=[0..486.8]
col row_id integer uniq=32391 range=[1000..3.339e+04]
col d binary uniq=2 values=treated|control
col feature_11 numeric uniq=6827 range=[0..108.7]
col feature_8 integer uniq=20 range=[0..19]
col feature_10 numeric uniq=4871 range=[0..6.949]
col feature_1 numeric uniq=4087 range=[0..61.07]
```

**Parse:** `<T:effect> treatment=`d` outcome=`Y` randomized=`no``

```
size rows=32391 covariates=12
overlap auc=0.638 extreme=0 maxsmd=0.333
flags good_overlap imbalance large_sample
allowed causal_forest dml doubly_robust ipw matching psm regression sensitivity_analysis
warning -
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

Several methods fit this data about equally well: Double machine learning and Doubly robust estimation (AIPW / DR-learner).
Covariates predict the treatment with AUC=0.638; 0% of units have extreme propensity scores.
- Double machine learning: DML shines on large samples
- Doubly robust estimation (AIPW / DR-learner): flexible ML models can be used in both parts
Assumptions to check: the ML models predict outcome and treatment reasonably well; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 2. backdoor_small (en, mode=recommend)

**Question:** quick question: does the smaller class size lead to higher graduation rate?

```
rows=564 cols=7 structure=cross_section
col previous_grade numeric uniq=185 range=[1.53..4.32]
col class_size integer uniq=23 range=[9..35]
col parent_income numeric uniq=564 range=[0..1.213e+05]
col small_class binary uniq=2 range=[0..1] values=1|0
col graduated_pct numeric uniq=557 range=[54.81..101.7]
col parent_education_years numeric uniq=541 range=[5.049..20.82]
col days_absent integer uniq=16 range=[0..16]
```

**Parse:** `<T:effect> treatment=`small_class` outcome=`graduated_pct``

```
size rows=564 covariates=5
overlap auc=0.756 extreme=0.00887 maxsmd=0.635
flags good_overlap imbalance
allowed ipw matching psm regression sensitivity_analysis
warning causal_forest dml doubly_robust diff_in_means
```

**Methods:** {'regression': 1.0, 'psm': 0.6, 'ipw': 0.5, 'matching': 0.6, 'doubly_robust': 0.4}

**Explanation:**

Use Linear regression with an adjustment set.
Regress the outcome on the treatment plus a set of confounders that blocks all back-door paths.
Covariates predict the treatment with AUC=0.756; 1% of units have extreme propensity scores.
The data has 564 rows.
Alternative: Propensity score matching, since propensity distributions of the groups overlap.
Not recommended: Difference in means (randomized experiment), because groups differ on covariates, so randomization is questionable.
Assumptions to check: no mediators or colliders are included as controls; the relationship is approximately linear (or correctly specified).

## 3. continuous_treatment (ru, mode=recommend)

**Question:** как иззмерить эфефкт impressions на `conversions` по этим данным?

```
rows=1520 cols=13 structure=cross_section
col platform categorical uniq=3 values=mobile|tablet|desktop
col tenure_months numeric uniq=1230 range=[0..53.69]
col historic_spend numeric uniq=1497 range=[0..762.7]
col cart_value numeric uniq=1379 range=[0..117.3]
col q2_score numeric uniq=1247 range=[0..6.225]
col impressions numeric uniq=1519 range=[120..5.702e+04]
col conversions numeric uniq=1508 range=[27.97..232.3]
col customer_age numeric uniq=1498 range=[3.533..72.29]
col subscribed binary uniq=2 values=no|yes
col income numeric uniq=1520 range=[1231..1.219e+05]
col idx1 numeric uniq=1343 range=[1.44..107.5]
col n_orders integer uniq=16 range=[0..15]
col acquisition_channel categorical uniq=4 values=direct|search|social|referral
```

**Parse:** `<T:effect> treatment=`impressions` outcome=`conversions``

```
size rows=1520 covariates=11
flags -
allowed regression sensitivity_analysis
warning causal_forest dml
```

**Methods:** {'dml': 1.0, 'causal_forest': 0.6, 'regression': 0.3}

**Explanation:**

Лучше всего подойдёт Двойное машинное обучение (DML).
ML-модели убирают влияние конфаундеров из исхода и воздействия, эффект оценивается по остаткам.
В данных 1520 строк.
Альтернатива: Причинный лес.
Проверьте допущения: все конфаундеры наблюдаемы; ML-модели хорошо предсказывают исход и воздействие.

## 4. discovery_gaussian (en, mode=options)

**Question:** What are the causal relationships between these variables? What approach would you recommend?

```
rows=2423 cols=9 structure=cross_section
col Age numeric uniq=1970 range=[10.92..20.96]
col Parent Education Years numeric uniq=2126 range=[4.504..22.04]
col Exam Score numeric uniq=1869 range=[25.14..109.4]
col Family Income numeric uniq=2423 range=[400..1.568e+05]
col Earnings numeric uniq=2423 range=[2137..8.3e+04]
col Graduated Pct numeric uniq=1567 range=[57.05..109.4]
col Previous Grade numeric uniq=259 range=[1.34..4.67]
col Absences integer uniq=19 range=[0..19] missing=3%
col School Type categorical uniq=3 values=charter|public|private
```

**Parse:** `<T:discovery>`

```
size rows=2423 covariates=9
normality nongaussian=0.125
flags gaussian_data
allowed ges fci notears pc
warning lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

There is no single best choice here; PC algorithm and Greedy equivalence search are both good options.
12% of the continuous variables are clearly non-Gaussian.
The data has 2423 rows.
- PC algorithm: Constraint-based discovery that removes edges using conditional independence tests and orients edges through v-structures.
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: faithfulness (independences in data come only from the graph); independent observations.

## 5. discovery_hidden (ru, mode=recommend)

**Question:** Какие из переменных являются причинами Возраст? Не все важные факторы есть в данных.

```
rows=3926 cols=10 structure=cross_section
col Index 1 numeric uniq=3789 range=[0.3..133.6]
col Стаж numeric uniq=3514 range=[0.1..39.6]
col Productivity numeric uniq=2921 range=[31.87..176.9]
col Commute Minutes numeric uniq=3776 range=[0.3..124.7]
col Зарплата numeric uniq=3926 range=[280..1.249e+05]
col Last Rating numeric uniq=424 range=[0.98..7.41]
col Текучесть numeric uniq=1693 range=[0.1..52.27]
col Возраст numeric uniq=3686 range=[0.2..113.7]
col Team Size integer uniq=24 range=[0..26] missing=6%
col Education categorical uniq=4 values=school|phd|bachelor|master
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=3926 covariates=10
normality nongaussian=0.778
flags non_gaussian
allowed fci
warning lingam ges notears pc
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

Лучше всего подойдёт Алгоритм FCI.
78% непрерывных переменных заметно негауссовские.
Почему подходит: есть подозрение на скрытый конфаундер.
Алгоритм PC здесь хуже: есть подозрение на скрытый конфаундер.
Считайте полученный граф гипотезой и сверьте его со знанием предметной области.
Проверьте допущения: faithfulness; наблюдения независимы.

## 6. discovery_nongaussian (en, mode=recommend)

**Question:** Learn the causal structure from the data, please. How should I analyse this

```
rows=5270 cols=7 structure=cross_section
col Systolic Bp numeric uniq=5004 range=[74.39..187.1]
col Days To Recovery numeric uniq=4428 range=[1.592..28.55]
col Patient Age numeric uniq=4997 range=[30.99..79.13]
col Readmission Rate numeric uniq=4905 range=[15.73..64.25]
col Cholesterol numeric uniq=4232 range=[138.2..522]
col Insurance categorical uniq=3 values=none|public|private
col Smoker binary uniq=2 values=no|yes
```

**Parse:** `<T:discovery>`

```
size rows=5270 covariates=7
normality nongaussian=0.8
flags large_sample non_gaussian
allowed lingam notears pc fci ges
warning -
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

Use LiNGAM.
Functional discovery that uses non-Gaussian noise in linear relationships to recover the full direction of every edge.
80% of the continuous variables are clearly non-Gaussian.
The data has 5270 rows.
Why it fits: non-Gaussian data lets LiNGAM orient every edge.
Alternative: PC algorithm, since independence tests are reliable on large samples.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: relationships are linear; noise terms are non-Gaussian.

## 7. front_door (en, mode=recommend)

**Question:** My manager asked me this: What's the best way to measure the effect of smoking cessation program (`cessation_program`) on `readmissions` with this data? I worry that unmeasured doctor preferences drives both smoking cessation program (`cessation_program`) and `readmissions`. The whole effect of smoking cessation program (`cessation_program`) on `readmissions` goes through adherence. What's the right methodology here?

```
rows=5666 cols=4 structure=cross_section
col patient_id integer uniq=5666 range=[1000..6665]
col cessation_program binary uniq=2 values=no|yes
col adherence numeric uniq=3728 range=[4.55..123.3]
col readmissions numeric uniq=2985 range=[0.2..76.6]
```

**Parse:** `<T:effect> treatment=`cessation_program` outcome=`readmissions` mediator=`adherence` hidden=`yes``

```
size rows=5666 covariates=1
overlap auc=0.502 extreme=0 maxsmd=0.0287
flags good_overlap large_sample randomization_plausible
allowed front_door sensitivity_analysis
warning diff_in_means causal_forest dml doubly_robust ipw matching psm regression
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Recommended method: Front-door adjustment.
If the whole effect flows through an observed mediator that is not confounded with the outcome, the effect is identified despite hidden treatment-outcome confounding.
Covariates predict the treatment with AUC=0.502; 0% of units have extreme propensity scores.
The data has 5666 rows.
Why it fits: front-door works where back-door adjustment is impossible.
Avoid: Inverse probability weighting, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: no unobserved confounding between treatment and mediator; all back-door paths from mediator to outcome are blocked by the treatment.

## 8. instrumental (ru, mode=recommend)

**Question:** увеличил ли `police_spending` `gdp_per_capita`? `neighbor_policy` влияет на `police_spending`, но на `gdp_per_capita` напрямую не влияет. боюсь, что неизмеренный фактор (местная экономическая ситуация) влияет и на `police_spending`, и на `gdp_per_capita`. как это правильно проанализировать?

```
rows=4227 cols=8 structure=cross_section
col police_spending numeric uniq=4068 range=[1.8..1116]
col безработица numeric uniq=904 range=[0..13.07]
col avg_price numeric uniq=3476 range=[0..215.1]
col age15to24 numeric uniq=127 range=[0.076..0.247]
col county_id integer uniq=4227 range=[1000..5226]
col neighbor_policy binary uniq=2 range=[0..1] values=1|0
col median_income numeric uniq=4227 range=[1.67e+04..7.951e+04]
col gdp_per_capita numeric uniq=4222 range=[1.571e+04..8.238e+04]
```

**Parse:** `<T:effect> treatment=`police_spending` outcome=`gdp_per_capita` instrument=`neighbor_policy` hidden=`yes``

```
size rows=4227 covariates=5
firststage F=1.82e+03
flags -
allowed iv sensitivity_analysis
warning causal_forest dml regression
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Лучше всего подойдёт Инструментальные переменные.
Инструмент сильный: F первой стадии = 1.82e+03.
В данных 4227 строк.
Почему подходит: есть подозрение на скрытый конфаундер.
Причинный лес здесь хуже: есть подозрение на скрытый конфаундер.
Проверьте допущения: инструмент действительно влияет на воздействие; инструмент не связан со скрытыми конфаундерами.

## 9. not_causal (en, mode=refuse)

**Question:** I'm analysing our retail data. Predict weekly_sales for new customers. What's the right methodology here?

```
rows=145 cols=6 structure=cross_section
col population numeric uniq=145 range=[5.691e+04..4.58e+05]
col parking_spaces integer uniq=75 range=[0..132]
col age numeric uniq=145 range=[8.94..75.49]
col weekly_sales numeric uniq=145 range=[4289..1.613e+04]
col past_purchases integer uniq=18 range=[0..18]
col in_loyalty binary uniq=2 values=False|True
```

**Parse:** `<T:not_causal> target=`weekly_sales``

```
size rows=145 covariates=6
flags small_sample
allowed -
warning -
```

**Methods:** {}

**Explanation:**

This is a prediction question, not a causal one: it asks how well `weekly_sales` can be predicted or described, not what happens if you change something. Use a predictive model (for example gradient boosting or regularised regression) with cross-validation, or descriptive statistics. Causal methods are needed only if you plan to intervene on a variable and want the effect of that change.

## 10. panel_factor (en, mode=recommend)

**Question:** need help: does `taxLaw` reduce crime rate (`crimesPer100k`)? any advice?

```
rows=1054 cols=5 structure=panel
panel unit=country(34) time=yr(31) balanced
col country integer uniq=34 range=[1..34]
col yr integer uniq=31 range=[1964..1994]
col crimesPer100k numeric uniq=1053 range=[2.4..1602]
col taxLaw binary uniq=2 values=False|True
col post_period binary uniq=2 range=[0..1] values=0|1
design group_post(taxLaw,post_period) treated=3 controls=31 pre=24 post=7 block
```

**Parse:** `<T:effect> treatment=`taxLaw` outcome=`crimesPer100k``

```
size rows=1054 covariates=0 treated=3 controls=31 pre=24 post=7
pretrend drift=606.0 rel=2.5 p=2.41e-19
scfit r2=0.131 donors=31 outside=0
flags has_controls policy_design pretrend_violated sc_poor_fit
allowed sdid sensitivity_analysis
warning synthetic_control did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.4, 'did': 0.1}

**Explanation:**

Best fit: Synthetic difference-in-differences.
Before the treatment the gap between treated and control units drifted by 606.0 (250% of the outcome's SD, p=2.41e-19), so the trends are not parallel.
A synthetic control fits the treated path before the treatment poorly (R²=0.131).
Why it fits: unit and time weights repair non-parallel trends.
Avoid: Difference-in-differences, because pre-treatment trends are not parallel, so DiD will be biased.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 11. panel_parallel (en, mode=recommend)

**Question:** Quick question: What is the effect of the minimum wage increase on `cigSales`? Which method should I use?

```
rows=264 cols=6 structure=panel
panel unit=country(33) time=yr(8) balanced
col country categorical uniq=33 values=state_1|state_2|state_3|state_4|state_5
col yr integer uniq=8 range=[1991..1998]
col cigSales numeric uniq=264 range=[24.83..216.6]
col wagePolicy binary uniq=2 range=[0..1] values=0|1
col age15to24 numeric uniq=77 range=[0.116..0.222]
col lnincome numeric uniq=264 range=[2.382e+04..7.761e+04]
design indicator(wagePolicy) treated=9 controls=24 pre=5 post=3 block
```

**Parse:** `<T:effect> treatment=`wagePolicy` outcome=`cigSales``

```
size rows=264 covariates=2 treated=9 controls=24 pre=5 post=3
pretrend drift=0.127 rel=0.00341 p=0.975
scfit r2=0.769 donors=5 outside=0
flags has_controls policy_design pretrend_ok
allowed did sdid sensitivity_analysis
warning synthetic_control regression
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

I recommend Difference-in-differences.
Compare the change in outcome before and after treatment in the treated group with the same change in the control group.
Pre-treatment trends look parallel (drift 0.127, p=0.975).
A synthetic control reproduces the treated path before the treatment well (R²=0.769).
Why it fits: pre-treatment trends of the groups are parallel.
Alternative: Synthetic difference-in-differences.
Avoid: Linear regression with an adjustment set, because with a treated-group and post-period design, plain regression ignores the time structure that DiD, SC and SDID exploit.
Assumptions to check: parallel trends (without treatment, both groups would have moved in parallel); no anticipation of the treatment.

## 12. panel_staggered (en, mode=recommend)

**Question:** hi! some companies got `treatment` and others didn't. did it change outcome?

```
rows=2424 cols=5 structure=panel
panel unit=unit_id(101) time=time(24) balanced
col unit_id integer uniq=101 range=[1..101]
col time datetime uniq=24 values=2019-04-01|2021-03-01
col outcome numeric uniq=2354 range=[0.1..58.18]
col treatment binary uniq=2 range=[0..1] values=1|0
col col_1 integer uniq=880 range=[0..1369] missing=4%
design indicator(treatment) treated=79 controls=22 pre=7 post=17 staggered
```

**Parse:** `<T:effect> treatment=`treatment` outcome=`outcome``

```
size rows=2424 covariates=1 treated=79 controls=22 pre=7 post=17
pretrend drift=-0.997 rel=0.168 p=0.0174
scfit r2=1 donors=9 outside=0
flags has_controls many_treated_units policy_design pretrend_violated sc_good_fit staggered
allowed sdid sensitivity_analysis
warning synthetic_control did dml regression
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Use Synthetic difference-in-differences.
Weighted DiD with unit weights (like synthetic control, but only trends must match, not levels) and time weights that focus on pre-periods most similar to the post-period.
Before the treatment the gap between treated and control units drifted by -0.997 (17% of the outcome's SD, p=0.0174), so the trends are not parallel.
A synthetic control reproduces the treated path before the treatment well (R²=1).
Why it fits: unit and time weights repair non-parallel trends; staggered adoption is handled by running SDID per adoption cohort and averaging.
Be careful with: Difference-in-differences, because pre-treatment trends are not parallel, so DiD will be biased.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 13. rct (ru, mode=recommend)

**Question:** Как оценить влияние free_shipping на purchase_amount? Free_shipping назначали случайно. Какой метод выбрать?

```
rows=6469 cols=11 structure=cross_section
col cart_value numeric uniq=4302 range=[0..129.4]
col tenure_months numeric uniq=3030 range=[0..55.09]
col acquisition_channel categorical uniq=4 values=search|social|referral|direct
col historic_spend numeric uniq=5971 range=[0..768.4]
col device categorical uniq=3 values=mobile|desktop|tablet
col past_orders integer uniq=18 range=[0..17]
col возраст numeric uniq=3324 range=[0..75.41]
col free_shipping binary uniq=2 range=[0..1] values=0|1
col purchase_amount numeric uniq=6253 range=[0.5..171.2]
col доход numeric uniq=6450 range=[0..1.215e+05]
col subscribed binary uniq=2 values=yes|no
```

**Parse:** `<T:effect> treatment=`free_shipping` outcome=`purchase_amount` randomized=`yes``

```
size rows=6469 covariates=9
overlap auc=0.492 extreme=0 maxsmd=0.0484
flags good_overlap large_sample randomization_plausible
allowed causal_forest diff_in_means dml doubly_robust ipw matching psm regression sensitivity_analysis
warning -
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

Рекомендуемый метод: Разность средних в эксперименте.
При случайном назначении эффект равен разнице средних исхода между группами.
Ковариаты предсказывают воздействие с AUC=0.492; у 0% объектов экстремальный propensity score.
Почему подходит: ковариаты сбалансированы, рандомизация выглядит правдоподобно.
Альтернатива: Линейная регрессия с контролем конфаундеров, так как ковариаты сбалансированы, рандомизация выглядит правдоподобно.
Проверьте допущения: воздействие действительно назначено случайно; нет выбывания участников, зависящего от группы.

## 14. rct_heterogeneous (en, mode=options)

**Question:** Who should we target with the email campaign to maximise `orders`? The email campaign was assigned at random

```
rows=1232 cols=9 structure=cross_section
col customer_id integer uniq=1232 range=[1000..2231]
col acquisition_channel categorical uniq=4 values=referral|social|direct|search
col subscribed binary uniq=2 values=no|yes
col tenure_months numeric uniq=1185 range=[0..51.75]
col device categorical uniq=3 values=mobile|tablet|desktop
col historic_spend numeric uniq=1143 range=[0..694.6] missing=7%
col email_sent binary uniq=2 values=control|treated
col orders numeric uniq=1167 range=[25.93..226]
col cart_value numeric uniq=1086 range=[0..119.5] missing=4%
```

**Parse:** `<T:heterogeneity> treatment=`email_sent` outcome=`orders` randomized=`yes``

```
size rows=1232 covariates=7
overlap auc=0.506 extreme=0 maxsmd=0.103
flags good_overlap randomization_plausible
allowed doubly_robust
warning meta_learners causal_forest dml
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

Several methods fit this data about equally well: Meta-learners (S-, T-, X-learner) and Causal forest (CausalForestDML).
Covariates predict the treatment with AUC=0.506; 0% of units have extreme propensity scores.
- Meta-learners (S-, T-, X-learner): on experimental data meta-learners give an honest estimate of heterogeneity
- Causal forest (CausalForestDML): An ensemble of causal trees that split the sample so that the effect is homogeneous within leaves.
Prefer Causal forest (CausalForestDML) over Meta-learners (S-, T-, X-learner) if provides confidence intervals for CATE.
Assumptions to check: no unobserved confounders (holds automatically in an experiment); SUTVA.

## 15. rdd (ru, mode=recommend)

**Question:** изменение `outcome` вызвано `treat` или это совпадение? спасибо!

```
rows=1334 cols=7 structure=cross_section
col treat binary uniq=2 range=[0..1] values=1|0
col col_2 categorical uniq=4 values=phd|school|master|bachelor
col col_4 integer uniq=685 range=[0..1236]
col outcome numeric uniq=899 range=[0.1..64.49]
col running numeric uniq=1315 range=[29.61..110.4]
col col_1 numeric uniq=1062 range=[1.184..6.483]
col col_3 numeric uniq=920 range=[0..25.6]
```

**Parse:** `<T:effect> treatment=`treat` outcome=`outcome``

```
size rows=1334 covariates=5
overlap auc=1 extreme=0.754 maxsmd=2.75
threshold var=running cutoff=64.9 acc=0.997 jump=0.974
flags imbalance poor_overlap running_variable_found sharp_cutoff
allowed rdd sensitivity_analysis
warning causal_forest diff_in_means dml doubly_robust ipw matching psm regression
```

**Methods:** {'rdd': 1.0}

**Explanation:**

Лучше всего подойдёт Разрывный регрессионный дизайн (RDD).
Воздействие включается при `running` >= 64.9: это правило объясняет 100% назначений.
Ковариаты предсказывают воздействие с AUC=1; у 75% объектов экстремальный propensity score.
В данных 1334 строк.
Почему подходит: воздействие назначено по порогу непрерывной переменной; порог определяет воздействие почти точно.
Взвешивание на обратную вероятность (IPW) здесь хуже: у части объектов вероятность воздействия почти 0 или почти 1.
Проверьте допущения: остальные факторы меняются у порога плавно; эффект оценивается только около порога.

## 16. time_series (en, mode=recommend)

**Question:** We only have one time series. How do we measure the effect of job training program (Обучение) on `Текучесть`?

```
rows=176 cols=6 structure=time_series
time_series time=неделя
col Team Size integer uniq=19 range=[0..19]
col неделя datetime uniq=176 values=2017-07-03|2020-11-09
col Текучесть numeric uniq=161 range=[12.76..24.51]
col Обучение binary uniq=2 range=[0..1] values=0|1
col Возраст numeric uniq=169 range=[25.08..53.53]
col Commute Minutes numeric uniq=164 range=[0..75.92]
```

**Parse:** `<T:effect> treatment=`Обучение` outcome=`Текучесть``

```
size rows=176 covariates=3
flags small_sample
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Recommended method: Interrupted time series / CausalImpact.
The data has 176 rows.
Assumptions to check: no other event coincides with the intervention; covariate series (if used) are not affected by the treatment.

## 17. unidentifiable (en, mode=refuse)

**Question:** i'm analysing our labor data. what happened to target because of `d`? but worker ambition probably affects both `d` and target, and we didn't measure it.

```
rows=466 cols=8 structure=cross_section
col var5 numeric uniq=230 range=[1.05..6.34] missing=7%
col var6 numeric uniq=383 range=[0..21.43]
col var3 categorical uniq=4 values=school|phd|bachelor|master
col var1 numeric uniq=460 range=[0..87.89]
col target numeric uniq=405 range=[0.13..31.47]
col var2 numeric uniq=461 range=[9.134..66.14]
col var4 categorical uniq=4 values=sales|finance|it|ops
col d binary uniq=2 values=False|True
```

**Parse:** `<T:effect> treatment=`d` outcome=`target` randomized=`no` hidden=`yes``

```
size rows=466 covariates=6
overlap auc=0.624 extreme=0 maxsmd=0.418
flags good_overlap imbalance
allowed sensitivity_analysis
warning ipw matching psm dml doubly_robust regression
```

**Methods:** {'sensitivity_analysis': 1.0}

**Explanation:**

The effect of `d` on `target` cannot be identified from this data. The unmeasured confounder you mention affects both, and there is no experiment, instrument or mediator to break that link, so regression, matching or DML would all be biased. The data has 466 rows. What you can do: run a sensitivity analysis to see how strong the hidden confounder would have to be to explain the association away; better, run an experiment, find a valid instrument, or measure the confounder.

## 18. backdoor_large (en, mode=options)

**Question:** hey, tobacco tax law -> crime rate: is there an effect?

```
rows=6905 cols=29 structure=cross_section
col fips integer uniq=6905 range=[1000..7904]
col lnincome numeric uniq=6897 range=[1.602e+04..8.129e+04]
col young_share numeric uniq=15 range=[0.1..0.24]
col urban_share numeric uniq=637 range=[0.28..1.201]
col unemployment numeric uniq=4456 range=[0..13.27]
col beer numeric uniq=1738 range=[8.42..38.64]
col avg_price numeric uniq=5101 range=[0..209.4]
col population numeric uniq=6863 range=[0..1.246e+07]
col q1_score numeric uniq=3231 range=[0..6.767]
col idx2 numeric uniq=6466 range=[0..112.8]
col idx3 numeric uniq=3998 range=[0..107.9]
col survey_q4 numeric uniq=544 range=[0..6.63]
col index_5 numeric uniq=3992 range=[0..101.8]
col q6_score numeric uniq=542 range=[0..6.35]
col idx7 numeric uniq=4078 range=[0..107.9]
col metric_8 numeric uniq=4834 range=[5.5..198.1]
col metric_9 numeric uniq=6648 range=[6.485..191.6]
col kpi_10 numeric uniq=4865 range=[0..184.2]
col survey_q11 numeric uniq=3239 range=[0..6.58]
col kpi_12 numeric uniq=6642 range=[14.85..192.3]
col survey_q13 numeric uniq=3230 range=[0..6.584]
col kpi_14 numeric uniq=6662 range=[6.392..193.6]
col index_15 numeric uniq=4009 range=[0..109.6]
col kpi_16 numeric uniq=4866 range=[20.09..187.4]
col kpi_17 numeric uniq=4885 range=[2.22..197.9]
col tax_law binary uniq=2 values=treated|control
col crime_rate numeric uniq=6298 range=[2.4..997]
col metric_18 numeric uniq=4829 range=[7.58..200.5]
col survey_q19 numeric uniq=3166 range=[0..7.033]
```

**Parse:** `<T:effect> treatment=`tax_law` outcome=`crime_rate``

```
size rows=6905 covariates=27
overlap auc=0.733 extreme=0.00377 maxsmd=0.744
flags good_overlap high_dim imbalance large_sample
allowed dml doubly_robust psm causal_forest ipw sensitivity_analysis
warning matching diff_in_means regression
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

There is no single best choice here; Double machine learning and Doubly robust estimation (AIPW / DR-learner) are both good options.
Covariates predict the treatment with AUC=0.733; 0% of units have extreme propensity scores.
- Double machine learning: DML shines on large samples
- Doubly robust estimation (AIPW / DR-learner): flexible ML models can be used in both parts
Assumptions to check: all confounders are observed; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 19. backdoor_small (en, mode=recommend)

**Question:** Hi! Is `tv_campaign` worth it in terms of `orders`? What approach would you recommend?

```
rows=85 cols=10 structure=cross_section
col platform categorical uniq=3 values=mobile|desktop|tablet
col household_income numeric uniq=85 range=[9850..9.657e+04]
col tv_campaign binary uniq=2 values=yes|no
col past_orders integer uniq=12 range=[0..11] missing=2%
col prev_spend numeric uniq=85 range=[0..525.9]
col acquisition_channel categorical uniq=4 values=search|referral|direct|social
col cart_value numeric uniq=85 range=[13.66..119]
col orders numeric uniq=85 range=[48.67..193.2]
col age numeric uniq=84 range=[15.03..65.59]
col tenure_months numeric uniq=81 range=[0..40.05] missing=1%
```

**Parse:** `<T:effect> treatment=`tv_campaign` outcome=`orders``

```
size rows=85 covariates=8
overlap auc=0.718 extreme=0.0824 maxsmd=0.842
flags imbalance small_sample
allowed sensitivity_analysis
warning regression matching diff_in_means
```

**Methods:** {'regression': 1.0, 'matching': 0.3}

**Explanation:**

Best fit: Linear regression with an adjustment set.
Covariates predict the treatment with AUC=0.718; 8% of units have extreme propensity scores.
Why it fits: on small samples a simple model is more stable than ML estimators.
Works best with at least 100 rows, has 85.
Be careful with: Difference in means (randomized experiment), because groups differ on covariates, so randomization is questionable.
Assumptions to check: all confounders are observed and included; no mediators or colliders are included as controls.

## 20. continuous_treatment (en, mode=recommend)

**Question:** Need help: What's the causal effect of years of education on test score? Thanks!

```
rows=2750 cols=14 structure=cross_section
col sex binary uniq=2 values=female|male
col school_type categorical uniq=3 values=charter|public|private
col previous_grade numeric uniq=266 range=[1.42..4.58]
col class_size integer uniq=28 range=[9..38]
col age numeric uniq=648 range=[10.65..21.26]
col family_income numeric uniq=2750 range=[0..1.292e+05]
col parent_education_years numeric uniq=959 range=[5.38..21.22]
col absences integer uniq=17 range=[0..17]
col kpi_1 numeric uniq=2365 range=[11.87..191.2]
col q2_score numeric uniq=1929 range=[0..6.512]
col schooling numeric uniq=1140 range=[2.06..27.7]
col math_score numeric uniq=2052 range=[29.08..115.4]
col q3_score numeric uniq=1941 range=[0..6.943]
col index_4 numeric uniq=2684 range=[0..100.8]
```

**Parse:** `<T:effect> treatment=`schooling` outcome=`math_score``

```
size rows=2750 covariates=12
flags -
allowed dml regression sensitivity_analysis
warning causal_forest
```

**Methods:** {'dml': 1.0, 'causal_forest': 0.6, 'regression': 0.3}

**Explanation:**

Recommended method: Double machine learning.
ML models partial out the influence of confounders from both the outcome and the treatment; the effect is estimated on the residuals with cross-fitting.
The data has 2750 rows.
Alternative: Causal forest (CausalForestDML).
Assumptions to check: all confounders are observed; the ML models predict outcome and treatment reasonably well.

## 21. discovery_gaussian (en, mode=options)

**Question:** Hi! How are `X1`, `X2` and X4 causally connected? Thanks!

```
rows=2571 cols=10 structure=cross_section
col X1 numeric uniq=2145 range=[39.22..159.5]
col X7 numeric uniq=2042 range=[0.3..104.5]
col X3 numeric uniq=397 range=[0.71..6.04]
col X5 numeric uniq=1874 range=[3.59..70.66]
col X4 numeric uniq=2571 range=[9572..9.687e+04]
col X6 numeric uniq=2394 range=[0.1..37.89]
col X2 numeric uniq=2409 range=[0.1..36.57]
col X9 categorical uniq=4 values=phd|bachelor|school|master
col X8 integer uniq=915 range=[0..1321]
col id integer uniq=2571 range=[1000..3570]
```

**Parse:** `<T:discovery>`

```
size rows=2571 covariates=10
normality nongaussian=0.222
flags gaussian_data
allowed ges fci notears pc
warning lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

Several methods fit this data about equally well: PC algorithm and Greedy equivalence search.
22% of the continuous variables are clearly non-Gaussian.
The data has 2571 rows.
- PC algorithm: Constraint-based discovery that removes edges using conditional independence tests and orients edges through v-structures.
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: causal sufficiency (no hidden common causes); independent observations.

## 22. discovery_hidden (en, mode=recommend)

**Question:** Hey, What drives what here? Not all relevant factrs are in the data.

```
rows=1528 cols=5 structure=cross_section
col Employment Rate numeric uniq=1416 range=[56.74..108]
col Median Income numeric uniq=1527 range=[3.272e+04..6.395e+04]
col Avg Price numeric uniq=1517 range=[0.6..320.4]
col Crimes Per 100k numeric uniq=1525 range=[2.4..897.8]
col Cigsale numeric uniq=1415 range=[15.49..257.3]
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=1528 covariates=5
normality nongaussian=1
flags non_gaussian
allowed fci
warning lingam ges notears pc
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

I recommend Fast causal inference (FCI).
An extension of PC that allows hidden confounders and marks edges that may be explained by them.
100% of the continuous variables are clearly non-Gaussian.
The data has 1528 rows.
Why it fits: designed for data with possible hidden confounders.
Avoid: LiNGAM, because LiNGAM assumes no hidden confounders.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 23. discovery_nongaussian (en, mode=recommend)

**Question:** need help: what drives what here? what approach would you recommend?

```
rows=936 cols=6 structure=cross_section
col Urban Share numeric uniq=59 range=[0.58..1.5]
col Gdp Pc numeric uniq=936 range=[1.091e+04..6.928e+04]
col Retprice numeric uniq=929 range=[0.6..240.7]
col Population numeric uniq=936 range=[4e+04..1.1e+07]
col Unemployment numeric uniq=854 range=[3.195..17.88]
col Young Share numeric uniq=104 range=[0.101..0.228]
```

**Parse:** `<T:discovery>`

```
size rows=936 covariates=6
normality nongaussian=0.833
flags non_gaussian
allowed -
warning lingam fci ges notears pc
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

Recommended method: LiNGAM.
Functional discovery that uses non-Gaussian noise in linear relationships to recover the full direction of every edge.
83% of the continuous variables are clearly non-Gaussian.
The data has 936 rows.
Why it fits: non-Gaussian data lets LiNGAM orient every edge.
Works best with at least 1000 rows, has 936.
Alternative: PC algorithm.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: relationships are linear; noise terms are non-Gaussian.

## 24. front_door (en, mode=recommend)

**Question:** I'm analysing our tech data. What's the best way to measure the effect of new onboarding flow on time in app with this data? I worry that unmeasured tech savviness drives both new onboarding flow and time in app. New onboarding flow changes `feature_usage`, and `feature_usage` is what changes time in app; nothing else connects them.

```
rows=5834 cols=4 structure=cross_section
col session_minutes numeric uniq=2645 range=[0.16..62.24]
col n_friends integer uniq=43 range=[0..45]
col variant_b binary uniq=2 range=[0..1] values=0|1
col feature_usage numeric uniq=4875 range=[0.08..31.17]
```

**Parse:** `<T:effect> treatment=`variant_b` outcome=`session_minutes` mediator=`feature_usage` hidden=`yes``

```
size rows=5834 covariates=1
overlap auc=0.489 extreme=0 maxsmd=0.00922
flags good_overlap large_sample randomization_plausible
allowed front_door sensitivity_analysis
warning diff_in_means causal_forest dml doubly_robust ipw matching psm regression
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Best fit: Front-door adjustment.
If the whole effect flows through an observed mediator that is not confounded with the outcome, the effect is identified despite hidden treatment-outcome confounding.
Covariates predict the treatment with AUC=0.489; 0% of units have extreme propensity scores.
Why it fits: front-door works where back-door adjustment is impossible.
Not recommended: Causal forest (CausalForestDML), because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: no unobserved confounding between treatment and mediator; all back-door paths from mediator to outcome are blocked by the treatment.

## 25. instrumental (en, mode=recommend)

**Question:** I'm analysing our tech data. How big is the effect of the number of notifications on retention? The random rollout bucket influences the number of notifications and is unrelated to everything else that affects retention. We don't observe tech savviness, which likely influences who gets the number of notifications and also retention. Any advice?

```
rows=21591 cols=6 structure=cross_section
col platform categorical uniq=3 values=ios|web|android
col device_price numeric uniq=21133 range=[0..1263]
col n_notifications numeric uniq=11083 range=[0.04..29.09]
col rollout_bucket binary uniq=2 range=[0..1] values=0|1
col account_age numeric uniq=19349 range=[0..441.7]
col retained numeric uniq=16250 range=[0.2..79.83]
```

**Parse:** `<T:effect> treatment=`n_notifications` outcome=`retained` instrument=`rollout_bucket` hidden=`yes``

```
size rows=21591 covariates=3
firststage F=9.71e+03
flags large_sample
allowed iv sensitivity_analysis
warning causal_forest dml regression
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Use Instrumental variables (2SLS).
The instrument is strong: first-stage F=9.71e+03.
Why it fits: IV is one of the few ways to handle unobserved confounders.
Not recommended: Double machine learning, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: relevance (the instrument really moves the treatment); monotonicity, for a LATE interpretation.

## 26. not_causal (ru, mode=refuse)

**Question:** Спрогнозируй `exam_score` на следующий период. Как это правильно проанализировать

```
rows=101 cols=6 structure=cross_section
col school_type categorical uniq=3 values=charter|public|private
col parent_education_years numeric uniq=92 range=[7.78..18.76]
col small_class binary uniq=2 range=[0..1] values=1|0
col previous_grade numeric uniq=79 range=[1.52..4.17]
col exam_score numeric uniq=100 range=[35.46..95.96]
col age numeric uniq=97 range=[12.02..21.11]
```

**Parse:** `<T:not_causal> target=`exam_score``

```
size rows=101 covariates=6
flags small_sample
allowed -
warning -
```

**Methods:** {}

**Explanation:**

Это вопрос о прогнозе, а не о причинности: важно, насколько хорошо можно предсказать или описать `exam_score`, а не что будет, если что-то изменить. Используйте предсказательную модель (например, градиентный бустинг или регуляризованную регрессию) с кросс-валидацией или описательную статистику. Причинные методы нужны, только если вы собираетесь вмешиваться и хотите знать эффект.

## 27. panel_factor (en, mode=options)

**Question:** Hello, We track `outputPerHour` for many companies over time. How did union membership affect it?

```
rows=360 cols=7 structure=panel
panel unit=employer(20) time=year(18) balanced
col employer categorical uniq=20 values=company_1|company_2|company_3|company_4|company_5
col unionMember binary uniq=2 values=no|yes
col year integer uniq=18 range=[2006..2023]
col teamSize integer uniq=20 range=[0..19]
col is_post binary uniq=2 values=no|yes
col outputPerHour numeric uniq=356 range=[26.82..185.6]
col department categorical uniq=4 values=finance|it|ops|sales
design group_post(unionMember,is_post) treated=1 controls=19 pre=12 post=6 block
```

**Parse:** `<T:effect> treatment=`unionMember` outcome=`outputPerHour``

```
size rows=360 covariates=2 treated=1 controls=19 pre=12 post=6
pretrend drift=23.3 rel=0.873 p=6.51e-05
scfit r2=0.958 donors=5 outside=0
flags has_controls policy_design pretrend_violated sc_good_fit single_treated_unit
allowed synthetic_control sdid sensitivity_analysis
warning did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.9, 'did': 0.1}

**Explanation:**

There is no single best choice here; Synthetic difference-in-differences and Synthetic control are both good options.
Before the treatment the gap between treated and control units drifted by 23.3 (87% of the outcome's SD, p=6.51e-05), so the trends are not parallel.
A synthetic control reproduces the treated path before the treatment well (R²=0.958).
- Synthetic difference-in-differences: unit and time weights repair non-parallel trends
- Synthetic control: designed for one or a few treated units
Prefer Synthetic difference-in-differences over Synthetic control if lower variance thanks to fixed effects, and levels need not match.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 28. panel_parallel (ru, mode=recommend)

**Question:** В части школ ввели `D`, в остальных нет. Изменился ли `Y`?

```
rows=264 cols=5 structure=panel
panel unit=unit_id(22) time=period(12) balanced
col var1 binary uniq=2 values=male|female
col period integer uniq=12 range=[1968..1979]
col unit_id integer uniq=22 range=[1..22]
col Y numeric uniq=257 range=[23.64..84.08]
col D binary uniq=2 values=no|yes
design indicator(var1) treated=22 controls=0 pre=0 post=12 staggered reversible
design indicator(D) treated=8 controls=14 pre=9 post=3 block
```

**Parse:** `<T:effect> treatment=`D` outcome=`Y``

```
size rows=264 covariates=0 treated=8 controls=14 pre=9 post=3
pretrend drift=-0.497 rel=0.0362 p=0.673
scfit r2=0.952 donors=6 outside=0
flags has_controls policy_design pretrend_ok sc_good_fit
allowed did sdid sensitivity_analysis
warning synthetic_control its regression
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

Рекомендуемый метод: Разность разностей (DiD).
Сравнивает изменение исхода до и после воздействия в группе с воздействием с тем же изменением в контрольной группе.
Тренды до воздействия выглядят параллельными (смещение -0.497, p=0.673).
Синтетический контроль хорошо повторяет траекторию до воздействия (R²=0.952).
Объектов с воздействием 8, контрольных 14; периодов до воздействия 9, после 3.
Почему подходит: тренды до воздействия параллельны.
Альтернатива: Синтетическая разность разностей (SDID).
Прерванные временные ряды / CausalImpact здесь хуже: есть контрольные объекты.
Проверьте допущения: параллельные тренды без воздействия; нет перетока эффекта на контрольные объекты.

## 29. panel_staggered (en, mode=recommend)

**Question:** Hello, Wge Poliyc was introduced in some states. What did it do to cigarette sales per capita? What's the right methodology here?

```
rows=1470 cols=6 structure=panel
panel unit=State Id(70) time=year(21) balanced
col Beer numeric uniq=935 range=[12.5..35.11] missing=4%
col Age15to24 numeric uniq=14 range=[0.11..0.24] missing=5%
col Cig Sales numeric uniq=1373 range=[43.51..307.6]
col State Id categorical uniq=70 values=state_1|state_2|state_3|state_4|state_5
col year integer uniq=21 range=[1969..1989]
col Wage Policy binary uniq=2 values=no|yes
design indicator(Wage Policy) treated=58 controls=12 pre=8 post=13 staggered
```

**Parse:** `<T:effect> treatment=`Wage Policy` outcome=`Cig Sales``

```
size rows=1470 covariates=2 treated=58 controls=12 pre=8 post=13
pretrend drift=-2.76 rel=0.0994 p=0.299
scfit r2=0.997 donors=7 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit staggered
allowed sdid sensitivity_analysis
warning did synthetic_control dml its regression
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Best fit: Synthetic difference-in-differences.
Pre-treatment trends look parallel (drift -2.76, p=0.299).
A synthetic control reproduces the treated path before the treatment well (R²=0.997).
There are 58 treated units and 12 control units, 8 periods before and 13 after the treatment.
Why it fits: staggered adoption is handled by running SDID per adoption cohort and averaging.
Be careful with: Difference-in-differences, because naive two-way fixed effects is biased when units are treated at different times.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 30. rct (en, mode=recommend)

**Question:** My manager asked me this: Can we say that the tutoring program improves earnings? How should I analyse this?

```
rows=4174 cols=6 structure=cross_section
col parent_income numeric uniq=4167 range=[0..1.283e+05]
col class_size integer uniq=27 range=[11..39]
col sex binary uniq=2 values=male|female
col school_type categorical uniq=3 values=public|charter|private
col tutoring categorical uniq=3 values=0|2|1
col earnings numeric uniq=4174 range=[6844..7.776e+04]
```

**Parse:** `<T:effect> treatment=`tutoring` outcome=`earnings``

```
size rows=4174 covariates=4
flags -
allowed causal_forest dml doubly_robust ipw regression sensitivity_analysis
warning diff_in_means
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

I recommend Difference in means (randomized experiment).
The data has 4174 rows.
Valid only if treatment was randomized; please confirm.
Alternative: Linear regression with an adjustment set.
Assumptions to check: treatment was genuinely randomized; no differential attrition between groups.

## 31. rct_heterogeneous (en, mode=options)

**Question:** Hey, For whom does union membership work best in terms of wage? How should I analyse this?

```
rows=8718 cols=11 structure=cross_section
col last_rating numeric uniq=3100 range=[0.481..6.24]
col experience_years numeric uniq=6423 range=[0..29.04]
col company_size integer uniq=1121 range=[0..1637]
col education categorical uniq=4 values=phd|school|master|bachelor
col team_size integer uniq=25 range=[0..24]
col commute_minutes numeric uniq=4428 range=[0..100.9]
col dept categorical uniq=4 values=sales|finance|ops|it
col union_member binary uniq=2 range=[0..1] values=0|1
col wage numeric uniq=8710 range=[280..1.044e+05]
col age numeric uniq=7750 range=[1.904..73.3]
col idx1 numeric uniq=4570 range=[0..112.3]
```

**Parse:** `<T:heterogeneity> treatment=`union_member` outcome=`wage``

```
size rows=8718 covariates=9
overlap auc=0.502 extreme=0 maxsmd=0.0593
flags good_overlap large_sample randomization_plausible
allowed meta_learners causal_forest dml doubly_robust
warning -
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

There is no single best choice here; Meta-learners (S-, T-, X-learner) and Causal forest (CausalForestDML) are both good options.
Covariates predict the treatment with AUC=0.502; 0% of units have extreme propensity scores.
The data has 8718 rows.
- Meta-learners (S-, T-, X-learner): on experimental data meta-learners give an honest estimate of heterogeneity
- Causal forest (CausalForestDML): enough data to find subgroups with different effects
Prefer Causal forest (CausalForestDML) over Meta-learners (S-, T-, X-learner) if provides confidence intervals for CATE.
Assumptions to check: overlap between groups on covariates (positivity); SUTVA.

## 32. rdd (en, mode=recommend)

**Question:** Ned help: Did `treat` actually work? I want to know its impact on `outcome`. Which method should I use?

```
rows=1248 cols=7 structure=cross_section
col feature_2 integer uniq=12 range=[0..11]
col treat binary uniq=2 values=no|yes
col feature_3 numeric uniq=1188 range=[0..32.35]
col id integer uniq=1248 range=[1000..2247]
col X numeric uniq=1175 range=[4..1390] missing=5%
col feature_1 integer uniq=49 range=[0..51]
col outcome numeric uniq=1246 range=[160.3..856.7]
```

**Parse:** `<T:effect> treatment=`treat` outcome=`outcome``

```
size rows=1248 covariates=5
overlap auc=0.994 extreme=0.7 maxsmd=2.51
threshold var=X cutoff=637.5 acc=0.998 jump=0.987
flags imbalance poor_overlap running_variable_found sharp_cutoff
allowed rdd sensitivity_analysis
warning causal_forest diff_in_means dml doubly_robust ipw matching psm regression
```

**Methods:** {'rdd': 1.0}

**Explanation:**

I recommend Regression discontinuity design.
When treatment is assigned by a threshold on a running variable, units just above and just below the threshold are comparable.
Treatment switches at `X` >= 637.5: this rule explains 100% of assignments.
Why it fits: treatment jumps at a threshold of a continuous variable; the threshold determines treatment almost perfectly (sharp RDD).
Not recommended: Linear regression with an adjustment set, because regression silently extrapolates where there are no comparable units.
Assumptions to check: units cannot precisely manipulate the running variable around the cutoff; other factors change smoothly at the cutoff.

## 33. time_series (en, mode=recommend)

**Question:** was the jump in `Y` caused by d? what's the right methodology here

```
rows=86 cols=5 structure=time_series
time_series time=t
col t datetime uniq=86 values=2020-09-08|2020-12-02
col D binary uniq=2 values=False|True
col Y numeric uniq=83 range=[32.92..44.41]
col var1 numeric uniq=86 range=[34.63..74.24]
col var2 numeric uniq=86 range=[16.53..40.6]
```

**Parse:** `<T:effect> treatment=`D` outcome=`Y``

```
size rows=86 covariates=2
flags small_sample
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Use Interrupted time series / CausalImpact.
The data has 86 rows.
Assumptions to check: the pre-treatment model extrapolates correctly into the post-period; no other event coincides with the intervention.

## 34. unidentifiable (en, mode=refuse)

**Question:** Hello, How should I analyse whether job training program (training) affects salary? But manager support probably affects both job training program (training) and salary, and we didn't measure it. This is observational data, job training program (training) wasn't assigned randomly. What's the right methodology here?

```
rows=1882 cols=3 structure=cross_section
col training binary uniq=2 values=no|yes
col wage numeric uniq=1882 range=[7284..9.919e+04]
col age numeric uniq=1722 range=[0.637..69.47] missing=6%
```

**Parse:** `<T:effect> treatment=`training` outcome=`wage` randomized=`no` hidden=`yes``

```
size rows=1882 covariates=1
overlap auc=0.599 extreme=0 maxsmd=0.365
flags good_overlap imbalance
allowed sensitivity_analysis
warning ipw matching psm causal_forest dml doubly_robust regression
```

**Methods:** {'sensitivity_analysis': 1.0}

**Explanation:**

The effect of `training` on `wage` cannot be identified from this data. The unmeasured confounder you mention affects both, and there is no experiment, instrument or mediator to break that link, so regression, matching or DML would all be biased. The data has 1882 rows. What you can do: run a sensitivity analysis to see how strong the hidden confounder would have to be to explain the association away; better, run an experiment, find a valid instrument, or measure the confounder.

## 35. backdoor_large (ru, mode=options)

**Question:** Сработал ли `препарат`? Хочу понять, как он повлиял на дни_восстановления. Какой метод выбрать?

```
rows=3749 cols=20 structure=cross_section
col survey_q5 numeric uniq=2346 range=[0..7.371]
col index_6 numeric uniq=3623 range=[0..97.66]
col comorbidities integer uniq=7 range=[0..6]
col bmi numeric uniq=1575 range=[10.07..44.26]
col возраст numeric uniq=3628 range=[0.797..101]
col baseline_score numeric uniq=2883 range=[0..11.65]
col q2_score numeric uniq=495 range=[0..6.52]
col metric_9 numeric uniq=3682 range=[15.3..192.1]
col survey_q4 numeric uniq=2367 range=[0..6.626]
col препарат binary uniq=2 range=[0..1] values=1|0
col kpi_8 numeric uniq=3064 range=[11.91..195.7]
col gender binary uniq=2 values=female|male
col q7_score numeric uniq=2333 range=[0..6.126]
col дни_восстановления numeric uniq=1453 range=[1.88..32.43]
col cholesterol numeric uniq=3685 range=[71.94..341.4]
col idx1 numeric uniq=3613 range=[0..102.2]
col survey_q3 numeric uniq=2383 range=[0..6.54]
col patient_id integer uniq=3749 range=[1000..4748]
col insurance categorical uniq=3 values=none|public|private
col smoker binary uniq=2 values=yes|no
```

**Parse:** `<T:effect> treatment=`препарат` outcome=`дни_восстановления` randomized=`no``

```
size rows=3749 covariates=18
overlap auc=0.61 extreme=0 maxsmd=0.221
flags good_overlap
allowed ipw matching psm causal_forest dml doubly_robust regression sensitivity_analysis
warning -
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

Здесь подходят несколько методов примерно одинаково хорошо: Двойное машинное обучение (DML) и Дважды робастная оценка.
Ковариаты предсказывают воздействие с AUC=0.61; у 0% объектов экстремальный propensity score.
В данных 3749 строк.
- Двойное машинное обучение (DML): ML-модели убирают влияние конфаундеров из исхода и воздействия, эффект оценивается по остаткам.
- Дважды робастная оценка: Сочетает модель исхода и модель propensity score; оценка верна, если верна хотя бы одна из них.
Проверьте допущения: все конфаундеры наблюдаемы; воздействие варьирует при любых значениях ковариат.

## 36. backdoor_small (en, mode=recommend)

**Question:** I'm analysing our marketing data. `treat` -> `Y`: is three an effect? How should I analyse this?

```
rows=171 cols=7 structure=cross_section
col col_1 numeric uniq=159 range=[0..615.9] missing=6%
col col_3 categorical uniq=4 values=social|search|direct|referral
col col_4 binary uniq=2 values=yes|no
col col_5 numeric uniq=159 range=[1.691e+04..1.048e+05] missing=7%
col treat binary uniq=2 range=[0..1] values=1|0
col Y numeric uniq=171 range=[21.54..168.4]
col col_2 categorical uniq=3 values=tablet|mobile|desktop
```

**Parse:** `<T:effect> treatment=`treat` outcome=`Y` randomized=`no``

```
size rows=171 covariates=5
overlap auc=0.774 extreme=0.0175 maxsmd=0.672
flags good_overlap imbalance small_sample
allowed regression sensitivity_analysis
warning ipw matching psm
```

**Methods:** {'regression': 1.0, 'psm': 0.6, 'ipw': 0.5, 'matching': 0.6}

**Explanation:**

I recommend Linear regression with an adjustment set.
Regress the outcome on the treatment plus a set of confounders that blocks all back-door paths.
Covariates predict the treatment with AUC=0.774; 2% of units have extreme propensity scores.
The data has 171 rows.
Why it fits: on small samples a simple model is more stable than ML estimators.
Alternative: Propensity score matching, since propensity distributions of the groups overlap.
Assumptions to check: no mediators or colliders are included as controls; the relationship is approximately linear (or correctly specified).

## 37. continuous_treatment (ru, mode=recommend)

**Question:** вопрос: как оценить влияние hours_studied на graduated_pct? спасибо!

```
rows=14701 cols=7 structure=cross_section
col class_size integer uniq=30 range=[11..40]
col graduated_pct numeric uniq=3114 range=[57.78..137.7]
col prior_gpa numeric uniq=324 range=[1.01..4.97]
col hours_studied numeric uniq=10109 range=[0.08..50.29]
col absences integer uniq=19 range=[0..18]
col student_id integer uniq=14701 range=[1000..1.57e+04]
col parent_education_years numeric uniq=1354 range=[2.76..23.23]
```

**Parse:** `<T:effect> treatment=`hours_studied` outcome=`graduated_pct``

```
size rows=14701 covariates=5
flags large_sample
allowed causal_forest dml regression sensitivity_analysis
warning -
```

**Methods:** {'dml': 1.0, 'causal_forest': 0.6, 'regression': 0.3}

**Explanation:**

Лучше всего подойдёт Двойное машинное обучение (DML).
ML-модели убирают влияние конфаундеров из исхода и воздействия, эффект оценивается по остаткам.
В данных 14701 строк.
Почему подходит: данных много.
Альтернатива: Причинный лес, так как данных много.
Проверьте допущения: все конфаундеры наблюдаемы; ML-модели хорошо предсказывают исход и воздействие.

## 38. discovery_gaussian (en, mode=options)

**Question:** Hello, Can you find the causal graph behind this data? What's the right methodology here?

```
rows=4953 cols=5 structure=cross_section
col Historic Spend numeric uniq=4926 range=[2.4..879.8]
col Cart Value numeric uniq=3594 range=[0.4..154.4]
col Tenure Months numeric uniq=2751 range=[0.2..71.84]
col Visits numeric uniq=4786 range=[163.8..1695]
col Age numeric uniq=4351 range=[3.702..80.03] missing=7%
```

**Parse:** `<T:discovery>`

```
size rows=4953 covariates=5
normality nongaussian=0
flags gaussian_data
allowed ges fci notears pc
warning lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

Several methods fit this data about equally well: PC algorithm and Greedy equivalence search.
0% of the continuous variables are clearly non-Gaussian.
- PC algorithm: Constraint-based discovery that removes edges using conditional independence tests and orients edges through v-structures.
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: causal sufficiency (no hidden common causes); independent observations.

## 39. discovery_hidden (en, mode=recommend)

**Question:** We have data on employees. Learn the causal structure from the data, please. Some common causes might be missing. What approach would you recommend?

```
rows=6501 cols=10 structure=cross_section
col f7 numeric uniq=6168 range=[31.17..202.9]
col f4 numeric uniq=6500 range=[30.06..8.564e+04]
col f3 numeric uniq=5902 range=[0.2..96.89]
col f5 numeric uniq=5492 range=[0.1..39.06]
col f8 numeric uniq=3860 range=[0.3..135.6]
col f1 numeric uniq=3635 range=[0.3..160.7]
col f6 numeric uniq=2624 range=[0.016..11.26]
col f2 numeric uniq=5402 range=[5.161..46.57]
col f9 integer uniq=25 range=[0..24]
col row_id integer uniq=6501 range=[1000..7500]
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=6501 covariates=10
normality nongaussian=1
flags large_sample non_gaussian
allowed fci
warning lingam notears pc ges
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

Best fit: Fast causal inference (FCI).
An extension of PC that allows hidden confounders and marks edges that may be explained by them.
100% of the continuous variables are clearly non-Gaussian.
The data has 6501 rows.
Why it fits: designed for data with possible hidden confounders.
Be careful with: Greedy equivalence search, because GES assumes no hidden confounders.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 40. discovery_nongaussian (ru, mode=recommend)

**Question:** Построй причинный граф по этим данным. Спасибо!

```
rows=773 cols=7 structure=cross_section
col Unemployment numeric uniq=735 range=[0.04..16.06]
col Cigsale numeric uniq=771 range=[17.99..223.7]
col Employment numeric uniq=756 range=[48.19..77.84]
col Young Share numeric uniq=108 range=[0.089..0.269]
col Beer numeric uniq=595 range=[9.46..33.49]
col Crimes Per 100k numeric uniq=773 range=[2.4..842.4]
col Median Income numeric uniq=773 range=[2.285e+04..7.503e+04]
```

**Parse:** `<T:discovery>`

```
size rows=773 covariates=7
normality nongaussian=0.429
flags gaussian_data
allowed -
warning ges fci notears pc lingam
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

Лучше всего подойдёт LiNGAM.
43% непрерывных переменных заметно негауссовские.
В данных 773 строк.
Альтернатива: Алгоритм PC.
Считайте полученный граф гипотезой и сверьте его со знанием предметной области.
Проверьте допущения: связи линейны; нет скрытых конфаундеров.

## 41. front_door (en, mode=recommend)

**Question:** I'd like to evaluate got_push. Did it change `retained`? I worry that unmeasured user engagement drives both got_push and `retained`. The whole effect of got_push on `retained` goes through feature usage. Thanks!

```
rows=6858 cols=6 structure=cross_section
col user_id integer uniq=6858 range=[1000..7857]
col prior_purchases integer uniq=8 range=[0..7]
col got_push binary uniq=2 values=control|treated
col feature_usage numeric uniq=1942 range=[0.08..31.31]
col retained numeric uniq=6261 range=[2.769..75.5]
col platform categorical uniq=3 values=android|web|ios
```

**Parse:** `<T:effect> treatment=`got_push` outcome=`retained` mediator=`feature_usage` hidden=`yes``

```
size rows=6858 covariates=3
overlap auc=0.498 extreme=0 maxsmd=0.0334
flags good_overlap large_sample randomization_plausible
allowed front_door sensitivity_analysis
warning diff_in_means causal_forest dml doubly_robust ipw matching psm regression
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Recommended method: Front-door adjustment.
Covariates predict the treatment with AUC=0.498; 0% of units have extreme propensity scores.
The data has 6858 rows.
Why it fits: front-door works where back-door adjustment is impossible.
Not recommended: Linear regression with an adjustment set, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: the mediator captures the entire effect of the treatment on the outcome; all back-door paths from mediator to outcome are blocked by the treatment.

## 42. instrumental (en, mode=recommend)

**Question:** We have data on customers. What's the best way to measure the effect of the discount rate on quantity with this data? We don't observe manager quality, which likely influences who gets the discount rate and also quantity. Distance to the distribution center (`dist_to_dc_km`) influences the discount rate and is unrelated to everything else that affects quantity. How should I analyse this?

```
rows=8617 cols=7 structure=cross_section
col discount_rate numeric uniq=7494 range=[0.1..70.46]
col customer_id integer uniq=8617 range=[1000..9616]
col n_employees integer uniq=55 range=[0..56] missing=7%
col dist_to_dc_km numeric uniq=4418 range=[0..93.03]
col quantity numeric uniq=7810 range=[34.11..974.1]
col floor_area numeric uniq=8271 range=[109.8..2286]
col parking_spaces integer uniq=147 range=[0..164] missing=5%
```

**Parse:** `<T:effect> treatment=`discount_rate` outcome=`quantity` instrument=`dist_to_dc_km` hidden=`yes``

```
size rows=8617 covariates=4
firststage F=2.57e+03
flags large_sample
allowed iv sensitivity_analysis
warning causal_forest dml regression
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Use Instrumental variables (2SLS).
The instrument is strong: first-stage F=2.57e+03.
The data has 8617 rows.
Why it fits: IV is one of the few ways to handle unobserved confounders.
Avoid: Double machine learning, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: exclusion restriction (the instrument affects the outcome only through the treatment); the instrument is as good as random (no confounding of instrument and outcome).

## 43. not_causal (ru, mode=refuse)

**Question:** Привет! Какая корреляция между n_orders и `конверсия`?

```
rows=562 cols=7 structure=cross_section
col конверсия numeric uniq=554 range=[7.96..213.8]
col device categorical uniq=3 values=mobile|desktop|tablet
col acquisition_channel categorical uniq=4 values=direct|search|social|referral
col n_orders integer uniq=15 range=[0..14]
col tv_on binary uniq=2 range=[0..1] values=0|1
col prev_spend numeric uniq=560 range=[0..705]
col возраст numeric uniq=522 range=[7.76..70.98]
```

**Parse:** `<T:not_causal> target=`конверсия``

```
size rows=562 covariates=7
flags -
allowed -
warning -
```

**Methods:** {}

**Explanation:**

Это вопрос о прогнозе, а не о причинности: важно, насколько хорошо можно предсказать или описать `конверсия`, а не что будет, если что-то изменить. Используйте предсказательную модель (например, градиентный бустинг или регуляризованную регрессию) с кросс-валидацией или описательную статистику. Причинные методы нужны, только если вы собираетесь вмешиваться и хотите знать эффект.

## 44. panel_factor (en, mode=options)

**Question:** does `lockdown` have a positive effect on cigarette sales per capita (`cigSales`)? any advice?

```
rows=1632 cols=7 structure=panel
panel unit=штат(48) time=год(34) balanced
col штат integer uniq=48 range=[1..48]
col lockdown binary uniq=2 range=[0..1] values=0|1
col urbanShare numeric uniq=75 range=[0.32..1.09]
col безработица numeric uniq=1471 range=[0..12.69]
col после binary uniq=2 range=[0..1] values=0|1
col год integer uniq=34 range=[1976..2009]
col cigSales numeric uniq=1539 range=[0.6..305.1]
design group_post(lockdown,после) treated=2 controls=46 pre=28 post=6 block
```

**Parse:** `<T:effect> treatment=`lockdown` outcome=`cigSales``

```
size rows=1632 covariates=2 treated=2 controls=46 pre=28 post=6
pretrend drift=63.0 rel=1.71 p=2.99e-13
scfit r2=0.0183 donors=46 outside=0
flags has_controls policy_design pretrend_violated sc_poor_fit
allowed sdid sensitivity_analysis
warning synthetic_control did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.9, 'did': 0.1}

**Explanation:**

Several methods fit this data about equally well: Synthetic difference-in-differences and Synthetic control.
Before the treatment the gap between treated and control units drifted by 63.0 (171% of the outcome's SD, p=2.99e-13), so the trends are not parallel.
- Synthetic difference-in-differences: unit and time weights repair non-parallel trends
- Synthetic control: does not require parallel trends, it matches the trajectory instead
Prefer Synthetic difference-in-differences over Synthetic control if lower variance thanks to fixed effects, and levels need not match.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 45. panel_parallel (en, mode=recommend)

**Question:** My manager asked me this: Is premium trial (`trial`) worth it in terms of retention? Any advice?

```
rows=343 cols=7 structure=panel
panel unit=country(49) time=month(7) balanced
col country integer uniq=49 range=[1..49]
col friends integer uniq=34 range=[0..35]
col trial binary uniq=2 values=control|treated
col userAge numeric uniq=342 range=[9.757..49.57]
col after_treatment binary uniq=2 values=yes|no
col month datetime uniq=7 values=2013-04-01|2013-10-01
col retentionD30 numeric uniq=343 range=[11.16..77.79]
design group_post(trial,after_treatment) treated=22 controls=27 pre=3 post=4 block
```

**Parse:** `<T:effect> treatment=`trial` outcome=`retentionD30``

```
size rows=343 covariates=2 treated=22 controls=27 pre=3 post=4
pretrend drift=-0.0813 rel=0.00613 p=0.734
scfit r2=-9.44 donors=1 outside=1
flags has_controls many_treated_units policy_design pretrend_ok sc_poor_fit short_pre_period treated_outside_hull
allowed did sensitivity_analysis
warning sdid dml regression synthetic_control
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

Recommended method: Difference-in-differences.
Pre-treatment trends look parallel (drift -0.0813, p=0.734).
Why it fits: pre-treatment trends of the groups are parallel; many treated units make the estimate precise.
Alternative: Synthetic difference-in-differences, since the intercept lets the treated unit differ in level from its synthetic control.
Avoid: Synthetic control, because the synthetic control does not reproduce the pre-treatment path.
Assumptions to check: parallel trends (without treatment, both groups would have moved in parallel); SUTVA (no spillovers to control units).

## 46. panel_staggered (en, mode=recommend)

**Question:** hi! compare stores with and without the discount campaign: what's the effect on `Sales`? how should i analyse this?

```
rows=816 cols=6 structure=panel
panel unit=Store(68) time=month(12) balanced
col Store categorical uniq=68 values=S001|S002|S003|S004|S005
col month datetime uniq=12 values=2014-08-01|2015-07-01
col Sales numeric uniq=816 range=[1497..1.763e+04]
col Discount Campaign binary uniq=2 range=[0..1] values=0|1
col Store Size Sqm numeric uniq=814 range=[215.2..2052]
col Store Age numeric uniq=659 range=[0..28.4]
design indicator(Discount Campaign) treated=57 controls=11 pre=4 post=8 staggered
```

**Parse:** `<T:effect> treatment=`Discount Campaign` outcome=`Sales``

```
size rows=816 covariates=2 treated=57 controls=11 pre=4 post=8
pretrend drift=388.9 rel=0.148 p=0.698
scfit r2=-5.19 donors=11 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_poor_fit short_pre_period staggered
allowed sensitivity_analysis
warning did sdid dml regression synthetic_control
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Recommended method: Synthetic difference-in-differences.
Pre-treatment trends look parallel (drift 388.9, p=0.698).
Why it fits: staggered adoption is handled by running SDID per adoption cohort and averaging.
Works best with at least 5 pre-treatment periods, has 4.
Avoid: Difference-in-differences, because naive two-way fixed effects is biased when units are treated at different times.
Assumptions to check: parallel trends after reweighting units and periods; no anticipation of the treatment.

## 47. rct (en, mode=recommend)

**Question:** How should I analyse whether union membership affects salary? We ran a randomized experiment. How should I analyse this?

```
rows=8277 cols=5 structure=cross_section
col dept categorical uniq=4 values=sales|ops|finance|it
col company_size integer uniq=1118 range=[0..1486]
col union_member categorical uniq=3 values=0|1|2
col annual_pay numeric uniq=8277 range=[2180..1.056e+05]
col experience_years numeric uniq=1827 range=[0..26.76]
```

**Parse:** `<T:effect> treatment=`union_member` outcome=`annual_pay` randomized=`yes``

```
size rows=8277 covariates=3
flags large_sample
allowed causal_forest dml doubly_robust diff_in_means ipw regression sensitivity_analysis
warning -
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

Best fit: Difference in means (randomized experiment).
The data has 8277 rows.
Alternative: Linear regression with an adjustment set.
Assumptions to check: SUTVA (one unit's treatment does not affect another unit's outcome); no differential attrition between groups.

## 48. rct_heterogeneous (en, mode=options)

**Question:** I want individual-level effects of the free laptop on earnings. It was a randomized A/B test.

```
rows=21001 cols=6 structure=cross_section
col absences integer uniq=20 range=[0..21]
col wage numeric uniq=20939 range=[240..1.041e+05]
col previous_grade numeric uniq=2569 range=[0.804..4.878]
col laptop_program binary uniq=2 values=control|treated
col parent_education_years numeric uniq=1437 range=[3.59..24.09]
col age numeric uniq=6178 range=[9.952..22.54]
```

**Parse:** `<T:heterogeneity> treatment=`laptop_program` outcome=`wage` randomized=`yes``

```
size rows=21001 covariates=4
overlap auc=0.502 extreme=0 maxsmd=0.026
flags good_overlap large_sample randomization_plausible
allowed meta_learners causal_forest dml doubly_robust
warning -
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

Several methods fit this data about equally well: Meta-learners (S-, T-, X-learner) and Causal forest (CausalForestDML).
Covariates predict the treatment with AUC=0.502; 0% of units have extreme propensity scores.
The data has 21001 rows.
- Meta-learners (S-, T-, X-learner): on experimental data meta-learners give an honest estimate of heterogeneity
- Causal forest (CausalForestDML): enough data to find subgroups with different effects
Prefer Causal forest (CausalForestDML) over Meta-learners (S-, T-, X-learner) if provides confidence intervals for CATE.
Assumptions to check: no unobserved confounders (holds automatically in an experiment); SUTVA.

## 49. rdd (en, mode=recommend)

**Question:** What is the effect of the smoking cessation program on days_to_recovery? The smoking cessation proggram was given to everyone with risk score (`triage_score`) of at least 93.4. What's the right methodology here?

```
rows=19012 cols=5 structure=cross_section
col chronic_conditions integer uniq=8 range=[0..7]
col triage_score numeric uniq=6784 range=[0.36..151.3]
col cholesterol numeric uniq=10153 range=[60.77..341.2]
col days_to_recovery numeric uniq=2022 range=[0.08..28]
col cessation_program binary uniq=2 values=no|yes
```

**Parse:** `<T:effect> treatment=`cessation_program` outcome=`days_to_recovery` running=`triage_score``

```
size rows=19012 covariates=3
overlap auc=0.879 extreme=0.195 maxsmd=1.53
threshold var=triage_score cutoff=93.3 acc=0.895 jump=0.776
flags imbalance large_sample poor_overlap running_variable_found
allowed rdd sensitivity_analysis
warning causal_forest dml doubly_robust diff_in_means ipw matching psm regression
```

**Methods:** {'rdd': 1.0}

**Explanation:**

Best fit: Regression discontinuity design.
Treatment switches at `triage_score` >= 93.3: this rule explains 90% of assignments.
Covariates predict the treatment with AUC=0.879; 20% of units have extreme propensity scores.
Why it fits: treatment jumps at a threshold of a continuous variable.
Be careful with: Propensity score matching, because scores are close to 0 or 1, there is nobody to match with.
Assumptions to check: units cannot precisely manipulate the running variable around the cutoff; other factors change smoothly at the cutoff.

## 50. time_series (en, mode=recommend)

**Question:** I have a CSV with 265 rows. What's the impact of Wage Policy on employment rate How should I analyse this?

```
rows=265 cols=5 structure=time_series
time_series time=day
col Retprice numeric uniq=259 range=[6.43..182.9]
col day datetime uniq=265 values=2020-03-02|2020-11-21
col Employment Rate numeric uniq=196 range=[57.11..65.4]
col Urban Share numeric uniq=56 range=[0.36..1.01]
col Wage Policy binary uniq=2 values=False|True
```

**Parse:** `<T:effect> treatment=`Wage Policy` outcome=`Employment Rate``

```
size rows=265 covariates=2
flags -
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Use Interrupted time series / CausalImpact.
Model the pre-treatment dynamics of the series (optionally with covariate series) and forecast the counterfactual after the intervention.
The data has 265 rows.
Assumptions to check: the pre-treatment model extrapolates correctly into the post-period; no other event coincides with the intervention.
