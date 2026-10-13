# SFT review sample

## 1. backdoor_large (en, mode=options)

**Question:** I hae a CSV with 32763 rows. What's the best way to measure the effect of remote worrk policy on `turnover_rate` with this data? What's the right methodology here?

```
rows=32763 cols=11 structure=cross_section
col q1_score numeric uniq=628 range=[0..7.13]
col experience numeric uniq=14409 range=[0..26.9]
col department categorical uniq=4 values=finance|sales|it|ops
col education categorical uniq=4 values=bachelor|school|phd|master
col wfh_policy binary uniq=2 range=[0..1] values=0|1
col company_size integer uniq=1248 range=[0..1496]
col team_size integer uniq=26 range=[0..25]
col turnover_rate numeric uniq=15074 range=[0.1..61.67]
col commute_minutes numeric uniq=24396 range=[0..98.31]
col last_rating numeric uniq=535 range=[0.2..6.82]
col age numeric uniq=21744 range=[0..81.43]
```

**Parse:** `<T:effect> treatment=`wfh_policy` outcome=`turnover_rate``

```
size rows=32763 covariates=9
overlap auc=0.642 extreme=0 maxsmd=0.335
nonlinearity r2_linear=0.359 r2_boosting=0.886
flags good_overlap imbalance large_sample nonlinear
allowed causal_forest dml doubly_robust ipw matching psm sensitivity_analysis
warning diff_in_means regression
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

There is no single best choice here; Double machine learning and Doubly robust estimation (AIPW / DR-learner) are both good options.
The outcome depends on the covariates non-linearly: boosting reaches R²=0.886 vs 0.359 for a linear model.
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

## 3. continuous_treatment (en, mode=recommend)

**Question:** Hi! Did `education_years` increase graduation rate? Any advice?

```
rows=1791 cols=5 structure=cross_section
col family_income numeric uniq=1789 range=[0..1.373e+05]
col parent_education_years numeric uniq=1631 range=[5.321..21.41]
col education_years numeric uniq=931 range=[3.16..23]
col graduated_pct numeric uniq=1743 range=[59.72..103.9]
col school_type categorical uniq=3 values=charter|private|public
```

**Parse:** `<T:effect> treatment=`education_years` outcome=`graduated_pct``

```
size rows=1791 covariates=3
nonlinearity r2_linear=0.755 r2_boosting=0.684
flags linear_ok
allowed regression sensitivity_analysis
warning causal_forest dml
```

**Methods:** {'regression': 1.0, 'dml': 0.8}

**Explanation:**

I recommend Linear regression with an adjustment set.
Regress the outcome on the treatment plus a set of confounders that blocks all back-door paths.
A linear model fits the outcome about as well as boosting (R²=0.755 vs 0.684).
Why it fits: relationships look linear, so a linear model is adequate.
Alternative: Double machine learning.
Assumptions to check: no mediators or colliders are included as controls; the relationship is approximately linear (or correctly specified).

## 4. discovery_gaussian (en, mode=options)

**Question:** I'm analysing our labor data. Can you find the causal graph behind this data? How should I analyse this?

```
rows=1171 cols=7 structure=cross_section
col Experience Years numeric uniq=1142 range=[0.1..34.08]
col Commute Minutes numeric uniq=1063 range=[0.3..98.33]
col Age numeric uniq=993 range=[0.2..80.02]
col Salary numeric uniq=1171 range=[5795..9.963e+04]
col Productivity numeric uniq=1160 range=[40.06..153.5]
col Turnover Rate numeric uniq=851 range=[0.1..34.73]
col Performance Score numeric uniq=335 range=[0.64..6.27]
```

**Parse:** `<T:discovery>`

```
size rows=1171 covariates=7
normality nongaussian=0
flags gaussian_data
allowed ges fci pc
warning notears lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

There is no single best choice here; PC algorithm and Greedy equivalence search are both good options.
0% of the continuous variables are clearly non-Gaussian.
The data has 1171 rows.
- PC algorithm: Constraint-based discovery that removes edges using conditional independence tests and orients edges through v-structures.
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: faithfulness (independences in data come only from the graph); independent observations.

## 5. discovery_hidden (en, mode=recommend)

**Question:** My manager asked me this: Dscover the causal structure of f6, f7 and `f2`. Some common causes might be missing. How should I analyse this?

```
rows=1265 cols=11 structure=cross_section
col f5 numeric uniq=1243 range=[0.2..86.5]
col f1 numeric uniq=1229 range=[1.6..873.3]
col f3 numeric uniq=1014 range=[0.16..78.18]
col f2 numeric uniq=1123 range=[0.3..154.4]
col f8 numeric uniq=1112 range=[0.3..110.3]
col f7 numeric uniq=1169 range=[0.5..226.2]
col f6 numeric uniq=1264 range=[4..1477]
col f4 numeric uniq=1008 range=[0.16..62.24]
col obs integer uniq=1265 range=[1000..2264]
col f10 integer uniq=8 range=[0..7]
col f9 categorical uniq=3 values=ios|web|android
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=1265 covariates=11
normality nongaussian=1
flags non_gaussian
allowed fci
warning lingam ges notears pc
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

Use Fast causal inference (FCI).
100% of the continuous variables are clearly non-Gaussian.
The data has 1265 rows.
Why it fits: designed for data with possible hidden confounders.
Avoid: Greedy equivalence search, because GES assumes no hidden confounders.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 6. discovery_nongaussian (ru, mode=recommend)

**Question:** Ессть данные, нужен совет. `Minutes In App` влиеят на `Subscriptions` или наоборот? Как это правильно проанализировать

```
rows=1151 cols=9 structure=cross_section
col Device Price numeric uniq=1141 range=[4..1647]
col Subscriptions numeric uniq=1139 range=[0.3..156.3]
col Minutes In App numeric uniq=923 range=[0.16..72.24]
col Kpi 1 numeric uniq=1081 range=[3.67..263.3]
col Удержание numeric uniq=991 range=[3.81..86.13]
col Возраст numeric uniq=896 range=[0.16..83.95]
col Days Since Signup numeric uniq=1114 range=[1.6..861.6]
col Prior Purchases integer uniq=7 range=[0..6]
col Country Tier categorical uniq=3 values=tier1|tier3|tier2
```

**Parse:** `<T:discovery>`

```
size rows=1151 covariates=9
normality nongaussian=1
flags non_gaussian
allowed lingam fci ges pc
warning notears
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

Лучше всего подойдёт LiNGAM.
Использует негауссовский шум в линейных связях, чтобы определить направление каждого ребра.
100% непрерывных переменных заметно негауссовские.
Почему подходит: данные заметно негауссовские.
Альтернатива: Алгоритм PC.
Считайте полученный граф гипотезой и сверьте его со знанием предметной области.
Проверьте допущения: шум негауссовский; нет скрытых конфаундеров.

## 7. front_door (en, mode=recommend)

**Question:** Hello, Quantify the effect of loyalty program on basket size. Loyalty program changes foot traffic, and foot traffic is what changes basket size; nothing else connects them. We don't observe local demand, which likely influences who gets loyalty program and also basket size. Which method should I use?

```
rows=10222 cols=6 structure=cross_section
col visitors numeric uniq=10209 range=[14..5570]
col parking_spaces integer uniq=143 range=[0..158] missing=2%
col basket_size numeric uniq=3190 range=[4.55..68.8]
col loyalty_member binary uniq=2 values=no|yes
col prior_orders integer uniq=23 range=[0..22]
col customer_age numeric uniq=9063 range=[0..87.31]
```

**Parse:** `<T:effect> treatment=`loyalty_member` outcome=`basket_size` mediator=`visitors` hidden=`yes``

```
size rows=10222 covariates=3
overlap auc=0.483 extreme=0 maxsmd=0.0118
nonlinearity r2_linear=0.119 r2_boosting=-0.00031
flags good_overlap large_sample linear_ok
allowed front_door sensitivity_analysis
warning causal_forest diff_in_means dml doubly_robust ipw matching psm regression
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

I recommend Front-door adjustment.
If the whole effect flows through an observed mediator that is not confounded with the outcome, the effect is identified despite hidden treatment-outcome confounding.
A linear model fits the outcome about as well as boosting (R²=0.119 vs -0.00031).
Why it fits: front-door works where back-door adjustment is impossible.
Not recommended: Linear regression with an adjustment set, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: no unobserved confounding between treatment and mediator; all back-door paths from mediator to outcome are blocked by the treatment.

## 8. instrumental (en, mode=recommend)

**Question:** I need to know whether `push_count` caused a change in subscriptions. But user engagement probably affects both `push_count` and subscriptions, and we didn't measure it. Random rollout bucket (rollout_bucket) influences `push_count` and is unrelated to everything else that affects subscriptions. Any advice?

```
rows=12854 cols=5 structure=cross_section
col rollout_bucket binary uniq=2 range=[0..1] values=1|0
col push_count numeric uniq=1665 range=[0.04..26.01]
col subscriptions numeric uniq=5310 range=[0.3..124.7]
col sessions_7d integer uniq=19 range=[0..18]
col country_tier categorical uniq=3 values=tier3|tier1|tier2
```

**Parse:** `<T:effect> treatment=`push_count` outcome=`subscriptions` instrument=`rollout_bucket` hidden=`yes``

```
size rows=12854 covariates=2
firststage F=3.24e+03
nonlinearity r2_linear=0.656 r2_boosting=0.597
flags large_sample linear_ok
allowed iv sensitivity_analysis
warning causal_forest dml regression
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

I recommend Instrumental variables (2SLS).
The instrument is strong: first-stage F=3.24e+03.
A linear model fits the outcome about as well as boosting (R²=0.656 vs 0.597).
The data has 12854 rows.
Why it fits: IV is one of the few ways to handle unobserved confounders.
Avoid: Causal forest (CausalForestDML), because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: exclusion restriction (the instrument affects the outcome only through the treatment); monotonicity, for a LATE interpretation.

## 9. not_causal (en, mode=refuse)

**Question:** Hello, Can you build a mdoel to predict graduation rate? Any advice?

```
rows=104 cols=7 structure=cross_section
col sex binary uniq=2 values=male|female
col family_income numeric uniq=101 range=[8578..1.205e+05] missing=3%
col prior_gpa numeric uniq=103 range=[1.701..4.041]
col days_absent integer uniq=14 range=[0..13] missing=2%
col class_size integer uniq=21 range=[13..36] missing=2%
col graduated_pct numeric uniq=104 range=[67.05..98.52]
col got_laptop binary uniq=2 values=no|yes
```

**Parse:** `<T:not_causal> target=`graduated_pct``

```
size rows=104 covariates=7
flags small_sample
allowed -
warning -
```

**Methods:** {}

**Explanation:**

This is a prediction question, not a causal one: it asks how well `graduated_pct` can be predicted or described, not what happens if you change something. Use a predictive model (for example gradient boosting or regularised regression) with cross-validation, or descriptive statistics. Causal methods are needed only if you plan to intervene on a variable and want the effect of that change.

## 10. panel_factor (en, mode=options)

**Question:** My manager asked me this: I'd like to evaluate free laptop (`laptopProgram`). Did it change graduation rate?

```
rows=2226 cols=5 structure=panel
panel unit=district(53) time=time(42) balanced
col laptopProgram binary uniq=2 values=no|yes
col district integer uniq=53 range=[1..53]
col classSize integer uniq=28 range=[12..40]
col graduatedPct numeric uniq=1620 range=[48.22..115.6]
col time integer uniq=42 range=[1..42]
design indicator(laptopProgram) treated=3 controls=50 pre=30 post=12 block
```

**Parse:** `<T:effect> treatment=`laptopProgram` outcome=`graduatedPct``

```
size rows=2226 covariates=1 treated=3 controls=50 pre=30 post=12
pretrend drift=-15.3 rel=1.59 p=1.5e-11
scfit r2=0.983 donors=10 outside=0
flags has_controls policy_design pretrend_violated sc_good_fit
allowed synthetic_control sdid sensitivity_analysis
warning did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.9, 'did': 0.1}

**Explanation:**

Several methods fit this data about equally well: Synthetic difference-in-differences and Synthetic control.
Before the treatment the gap between treated and control units drifted by -15.3 (159% of the outcome's SD, p=1.5e-11), so the trends are not parallel.
- Synthetic difference-in-differences: unit and time weights repair non-parallel trends
- Synthetic control: does not require parallel trends, it matches the trajectory instead
Why Synthetic difference-in-differences may be better than Synthetic control: lower variance thanks to fixed effects, and levels need not match.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 11. panel_parallel (en, mode=recommend)

**Question:** Need help: Is there a causal impact of free laptop (`gotLaptop`) on graduation rate? Which method should I use?

```
rows=1092 cols=5 structure=panel
panel unit=school(84) time=time(13) balanced
col gotLaptop binary uniq=2 range=[0..1] values=0|1
col school categorical uniq=84 values=school_1|school_2|school_3|school_4|school_5
col absences integer uniq=17 range=[0..16]
col graduatedPct numeric uniq=956 range=[42.69..103.8]
col time integer uniq=13 range=[1..13]
design indicator(gotLaptop) treated=22 controls=62 pre=7 post=6 block
```

**Parse:** `<T:effect> treatment=`gotLaptop` outcome=`graduatedPct``

```
size rows=1092 covariates=1 treated=22 controls=62 pre=7 post=6
pretrend drift=-0.161 rel=0.0178 p=0.8
scfit r2=1 donors=13 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit
allowed did sdid sensitivity_analysis
warning synthetic_control dml regression
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

I recommend Difference-in-differences.
Compare the change in outcome before and after treatment in the treated group with the same change in the control group.
Pre-treatment trends look parallel (drift -0.161, p=0.8).
Why it fits: pre-treatment trends of the groups are parallel; many treated units make the estimate precise.
Alternative: Synthetic difference-in-differences.
Be careful with: Synthetic control, because with many treated units, DiD or SDID is usually more efficient.
Assumptions to check: parallel trends (without treatment, both groups would have moved in parallel); no anticipation of the treatment.

## 12. panel_staggered (en, mode=recommend)

**Question:** I have a CSV wtih 840 rows. Help me figure out if the tutoring program has any effect on earnings. How should I analyse this

```
rows=840 cols=5 structure=panel
panel unit=district(84) time=t(10) balanced
col district integer uniq=84 range=[1..84]
col earnings numeric uniq=840 range=[240..7.568e+04]
col t integer uniq=10 range=[1..10]
col tutoring binary uniq=2 values=False|True
col absences integer uniq=16 range=[0..15]
design indicator(tutoring) treated=61 controls=23 pre=5 post=5 staggered
```

**Parse:** `<T:effect> treatment=`tutoring` outcome=`earnings``

```
size rows=840 covariates=1 treated=61 controls=23 pre=5 post=5
pretrend drift=-605.9 rel=0.0477 p=0.618
scfit r2=0.798 donors=23 outside=0
flags has_controls many_treated_units policy_design pretrend_ok staggered
allowed sdid sensitivity_analysis
warning did dml regression synthetic_control
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Recommended method: Synthetic difference-in-differences.
Pre-treatment trends look parallel (drift -605.9, p=0.618).
Why it fits: staggered adoption is handled by running SDID per adoption cohort and averaging.
Be careful with: Difference-in-differences, because naive two-way fixed effects is biased when units are treated at different times.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 13. rct (en, mode=recommend)

**Question:** Hi! Can we say that `small_class` improves graduation rate? Any advice?

```
rows=1684 cols=5 structure=cross_section
col small_class binary uniq=2 values=control|treated
col parent_education_years numeric uniq=794 range=[2.42..22.13]
col age numeric uniq=584 range=[9.51..20.67]
col class_size integer uniq=27 range=[11..39]
col graduation_rate numeric uniq=1253 range=[57.07..105.7]
```

**Parse:** `<T:effect> treatment=`small_class` outcome=`graduation_rate``

```
size rows=1684 covariates=3
overlap auc=0.5 extreme=0 maxsmd=0.0648
nonlinearity r2_linear=0.164 r2_boosting=-0.0566
flags good_overlap linear_ok randomization_plausible
allowed regression ipw matching psm doubly_robust sensitivity_analysis
warning diff_in_means causal_forest dml
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

Recommended method: Difference in means (randomized experiment).
When treatment is randomly assigned, the average effect is simply the difference in mean outcomes between the groups.
A linear model fits the outcome about as well as boosting (R²=0.164 vs -0.0566).
Covariates predict the treatment with AUC=0.5; 0% of units have extreme propensity scores.
Why it fits: covariates are balanced, so randomization looks credible.
Valid only if treatment was randomized; please confirm.
Alternative: Linear regression with an adjustment set, since relationships look linear, so a linear model is adequate.
Assumptions to check: treatment was genuinely randomized; SUTVA (one unit's treatment does not affect another unit's outcome).

## 14. rct_heterogeneous (en, mode=options)

**Question:** My manager asked me this: Which counties benefit most from the minimum wage increase? Assignment to the minimum wage increase was random

```
rows=9397 cols=5 structure=cross_section
col employment_rate numeric uniq=2219 range=[43.28..83.04]
col beer numeric uniq=1822 range=[9.92..38.97] missing=6%
col wage_policy binary uniq=2 range=[0..1] values=0|1
col avg_price numeric uniq=6376 range=[0..227.6]
col lnincome numeric uniq=9383 range=[1.322e+04..8.528e+04]
```

**Parse:** `<T:heterogeneity> treatment=`wage_policy` outcome=`employment_rate` randomized=`yes``

```
size rows=9397 covariates=3
overlap auc=0.505 extreme=0 maxsmd=0.0347
nonlinearity r2_linear=0.369 r2_boosting=0.386
flags good_overlap large_sample linear_ok randomization_plausible
allowed meta_learners causal_forest dml doubly_robust
warning -
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

Several methods fit this data about equally well: Meta-learners (S-, T-, X-learner) and Causal forest (CausalForestDML).
A linear model fits the outcome about as well as boosting (R²=0.369 vs 0.386).
Covariates predict the treatment with AUC=0.505; 0% of units have extreme propensity scores.
The data has 9397 rows.
- Meta-learners (S-, T-, X-learner): on experimental data meta-learners give an honest estimate of heterogeneity
- Causal forest (CausalForestDML): enough data to find subgroups with different effects
Why Causal forest (CausalForestDML) may be better than Meta-learners (S-, T-, X-learner): provides confidence intervals for CATE.
Assumptions to check: no unobserved confounders (holds automatically in an experiment); SUTVA.

## 15. rdd (en, mode=recommend)

**Question:** My manager asked me this: Did новый_онбординг increase time in app? Only users with `activity_score` above 72.3 could get новый_онбординг. How should I analyse this?

```
rows=4037 cols=4 structure=cross_section
col minutes_in_app numeric uniq=2223 range=[0.16..66.88]
col activity_score numeric uniq=3913 range=[0.4..138.2]
col user_id integer uniq=3750 range=[1000..5036] missing=7%
col новый_онбординг binary uniq=2 values=control|treated
```

**Parse:** `<T:effect> treatment=`новый_онбординг` outcome=`minutes_in_app` running=`activity_score``

```
size rows=4037 covariates=2
overlap auc=0.901 extreme=0.215 maxsmd=1.65
threshold var=activity_score cutoff=72.4 acc=0.9 jump=0.781
nonlinearity r2_linear=0.165 r2_boosting=0.145
flags imbalance linear_ok poor_overlap running_variable_found
allowed rdd sensitivity_analysis
warning regression causal_forest diff_in_means dml doubly_robust ipw matching psm
```

**Methods:** {'rdd': 1.0}

**Explanation:**

Use Regression discontinuity design.
When treatment is assigned by a threshold on a running variable, units just above and just below the threshold are comparable.
Treatment switches at `activity_score` >= 72.4: this rule explains 90% of assignments.
A linear model fits the outcome about as well as boosting (R²=0.165 vs 0.145).
Why it fits: treatment jumps at a threshold of a continuous variable.
Not recommended: Inverse probability weighting, because extreme weights make the estimate unstable.
Assumptions to check: units cannot precisely manipulate the running variable around the cutoff; other factors change smoothly at the cutoff.

## 16. time_series (en, mode=recommend)

**Question:** Need help: Help me figure out if the smaller class size has any effect on earnings (Earnings). What's the right methodology here?

```
rows=115 cols=6 structure=time_series
time_series time=period
col period datetime uniq=115 values=2007-11-01|2017-05-01
col Small Class binary uniq=2 values=False|True
col Earnings numeric uniq=115 range=[3.692e+04..6.634e+04]
col Parent Income numeric uniq=114 range=[3.696e+04..9.586e+04] missing=1%
col Parent Education Years numeric uniq=115 range=[9.632..17.33]
col Class Size integer uniq=19 range=[14..34]
```

**Parse:** `<T:effect> treatment=`Small Class` outcome=`Earnings``

```
size rows=115 covariates=3 pre=79 post=36
flags no_controls
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Use Interrupted time series / CausalImpact.
Model the pre-treatment dynamics of the series (optionally with covariate series) and forecast the counterfactual after the intervention.
This is a single series with no control group: 79 periods before the intervention and 36 after.
Why it fits: there is no control group, so a time-series counterfactual is the main option.
Assumptions to check: the pre-treatment model extrapolates correctly into the post-period; covariate series (if used) are not affected by the treatment.

## 17. unidentifiable (en, mode=refuse)

**Question:** Quick question: Does telemedicine program have a positive effect on readmissions (`readmission_rate`)? Patients opted into telemedicine program on their own. But patients' health awareness probably affects both telemedicine program and readmissions (`readmission_rate`), and we didn't measure it. What's the right methodology here?

```
rows=2727 cols=4 structure=cross_section
col patient_id integer uniq=2727 range=[1000..3726]
col sex binary uniq=2 values=female|male
col telehealth binary uniq=2 values=False|True
col readmission_rate numeric uniq=2615 range=[3.821..79.13]
```

**Parse:** `<T:effect> treatment=`telehealth` outcome=`readmission_rate` randomized=`no` hidden=`yes``

```
size rows=2727 covariates=2
overlap auc=0.563 extreme=0 maxsmd=0.284
nonlinearity r2_linear=0.29 r2_boosting=0.225
flags good_overlap imbalance linear_ok
allowed sensitivity_analysis
warning ipw matching psm regression causal_forest dml doubly_robust
```

**Methods:** {'sensitivity_analysis': 1.0}

**Explanation:**

The effect of `telehealth` on `readmission_rate` cannot be identified from this data. The unmeasured confounder you mention affects both, and there is no experiment, instrument or mediator to break that link, so regression, matching or DML would all be biased. The data has 2727 rows. What you can do: run a sensitivity analysis to see how strong the hidden confounder would have to be to explain the association away; better, run an experiment, find a valid instrument, or measure the confounder.

## 18. backdoor_large (en, mode=options)

**Question:** Hello, Did introducing new onboarding flow (`treatment_group`) make a difference to time in app? Any advice?

```
rows=13600 cols=14 structure=cross_section
col friends integer uniq=47 range=[0..47]
col prior_purchases integer uniq=8 range=[0..7]
col age numeric uniq=3466 range=[0..58.73]
col device_price numeric uniq=13403 range=[0..1293]
col days_since_signup numeric uniq=10051 range=[0..461.6]
col country_tier categorical uniq=3 values=tier1|tier2|tier3
col os categorical uniq=3 values=ios|web|android
col prev_sessions integer uniq=20 range=[0..19]
col idx1 numeric uniq=5361 range=[0..103.5]
col idx2 numeric uniq=11998 range=[0..105.3]
col treatment_group binary uniq=2 values=control|treated
col minutes_in_app numeric uniq=10720 range=[0.16..78.61]
col metric_3 numeric uniq=7246 range=[0..228.2]
col metric_4 numeric uniq=7254 range=[0..202.8]
```

**Parse:** `<T:effect> treatment=`treatment_group` outcome=`minutes_in_app` randomized=`no``

```
size rows=13600 covariates=12
overlap auc=0.625 extreme=0 maxsmd=0.246
nonlinearity r2_linear=0.401 r2_boosting=0.777
flags good_overlap large_sample nonlinear
allowed causal_forest dml doubly_robust ipw matching psm sensitivity_analysis
warning regression
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

Several methods fit this data about equally well: Double machine learning and Doubly robust estimation (AIPW / DR-learner).
The outcome depends on the covariates non-linearly: boosting reaches R²=0.777 vs 0.401 for a linear model.
Covariates predict the treatment with AUC=0.625; 0% of units have extreme propensity scores.
The data has 13600 rows.
- Double machine learning: flexible ML models capture the non-linear relationships found in the data
- Doubly robust estimation (AIPW / DR-learner): flexible models in both parts capture the non-linear relationships
Assumptions to check: the ML models predict outcome and treatment reasonably well; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 19. backdoor_small (ru, mode=recommend)

**Question:** Вопрос: Повлиял ли `union_member` на salary? `union_member` не рандомизировали, люди выбирали сами. Какой метод выбрать?

```
rows=139 cols=8 structure=cross_section
col commute_minutes numeric uniq=137 range=[0..76.51]
col team_size integer uniq=18 range=[0..17]
col company_size integer uniq=124 range=[0..1144]
col experience numeric uniq=136 range=[0..19.88]
col dept categorical uniq=4 values=sales|it|finance|ops
col union_member binary uniq=2 values=yes|no
col education categorical uniq=4 values=bachelor|school|phd|master
col salary numeric uniq=139 range=[1.572e+04..8.13e+04]
```

**Parse:** `<T:effect> treatment=`union_member` outcome=`salary` randomized=`no``

```
size rows=139 covariates=6
overlap auc=0.773 extreme=0.0647 maxsmd=0.685
nonlinearity r2_linear=0.733 r2_boosting=0.503
flags imbalance linear_ok small_sample
allowed regression sensitivity_analysis
warning ipw matching psm
```

**Methods:** {'regression': 1.0, 'psm': 0.6, 'ipw': 0.5, 'matching': 0.3}

**Explanation:**

Рекомендуемый метод: Линейная регрессия с контролем конфаундеров.
Линейная модель описывает исход почти так же хорошо, как бустинг (R²=0.733 против 0.503).
Почему подходит: связи близки к линейным; данных мало.
Альтернатива: Мэтчинг по propensity score.
Проверьте допущения: все конфаундеры наблюдаемы и включены в модель; связь близка к линейной.

## 20. continuous_treatment (en, mode=recommend)

**Question:** Hey, The study hours -> graduation_rate: is there an effect? What approach would you recommend

```
rows=2477 cols=11 structure=cross_section
col index_1 numeric uniq=1962 range=[0..102.8]
col class_size integer uniq=27 range=[11..37]
col graduation_rate numeric uniq=1563 range=[61.02..111.6]
col age numeric uniq=641 range=[10.31..21.98]
col days_absent integer uniq=17 range=[0..16]
col school_type categorical uniq=3 values=charter|public|private
col hours_studied numeric uniq=2322 range=[0.08..34.56]
col previous_grade numeric uniq=262 range=[1.38..4.51]
col family_income numeric uniq=2477 range=[0..1.216e+05]
col parent_education_years numeric uniq=945 range=[4.92..21.14]
col gender binary uniq=2 values=female|male
```

**Parse:** `<T:effect> treatment=`hours_studied` outcome=`graduation_rate``

```
size rows=2477 covariates=9
nonlinearity r2_linear=0.784 r2_boosting=0.844
flags nonlinear
allowed dml sensitivity_analysis
warning causal_forest regression
```

**Methods:** {'dml': 1.0, 'causal_forest': 0.6, 'regression': 0.3}

**Explanation:**

Recommended method: Double machine learning.
ML models partial out the influence of confounders from both the outcome and the treatment; the effect is estimated on the residuals with cross-fitting.
The outcome depends on the covariates non-linearly: boosting reaches R²=0.844 vs 0.784 for a linear model.
The data has 2477 rows.
Why it fits: flexible ML models capture the non-linear relationships found in the data.
Alternative: Causal forest (CausalForestDML), since trees capture non-linear relationships automatically.
Be careful with: Linear regression with an adjustment set, because the outcome depends on the covariates non-linearly, so a linear model is misspecified.
Assumptions to check: the ML models predict outcome and treatment reasonably well; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 21. discovery_gaussian (en, mode=options)

**Question:** need help: what drives what here? which method should i use?

```
rows=9396 cols=5 structure=cross_section
col feature_3 numeric uniq=2029 range=[12.42..44.04]
col feature_2 numeric uniq=6680 range=[78.56..328]
col feature_4 numeric uniq=8582 range=[73.89..180.5]
col feature_1 numeric uniq=8553 range=[0.28..113.3]
col feature_5 integer uniq=7 range=[0..6] missing=5%
```

**Parse:** `<T:discovery>`

```
size rows=9396 covariates=5
normality nongaussian=0
flags gaussian_data large_sample
allowed ges notears pc fci
warning lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

There is no single best choice here; PC algorithm and Greedy equivalence search are both good options.
0% of the continuous variables are clearly non-Gaussian.
The data has 9396 rows.
- PC algorithm: independence tests are reliable on large samples
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: causal sufficiency (no hidden common causes); independent observations.

## 22. discovery_hidden (en, mode=recommend)

**Question:** i have a csv with 2010 rows. i don't have a hypothesis yet. what causes what among site visits and tenure in months? some common causes might be missing. how should i analyse this?

```
rows=2010 cols=5 structure=cross_section
col Cart Value numeric uniq=1977 range=[21.66..147.8]
col Sessions numeric uniq=1985 range=[4..1729]
col Income numeric uniq=2010 range=[360..1.428e+05]
col Months Active numeric uniq=1959 range=[0.2..79.51]
col Historic Spend numeric uniq=1969 range=[2.4..1177]
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=2010 covariates=5
normality nongaussian=1
flags non_gaussian
allowed fci
warning lingam ges notears pc
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

Best fit: Fast causal inference (FCI).
100% of the continuous variables are clearly non-Gaussian.
The data has 2010 rows.
Why it fits: designed for data with possible hidden confounders.
Not recommended: LiNGAM, because LiNGAM assumes no hidden confounders.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 23. discovery_nongaussian (en, mode=recommend)

**Question:** Hello, What drives what here? How should I analyse this?

```
rows=6662 cols=5 structure=cross_section
col Output Per Hour numeric uniq=4185 range=[29.47..209.6]
col Attrition numeric uniq=1975 range=[0.1..61.68]
col Experience Years numeric uniq=2019 range=[0.1..60.82]
col Commute Minutes numeric uniq=6252 range=[0.3..136.2]
col Age numeric uniq=6110 range=[0..72.7]
```

**Parse:** `<T:discovery>`

```
size rows=6662 covariates=5
normality nongaussian=0.8
flags large_sample non_gaussian
allowed lingam notears pc fci ges
warning -
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

I recommend LiNGAM.
80% of the continuous variables are clearly non-Gaussian.
The data has 6662 rows.
Why it fits: non-Gaussian data lets LiNGAM orient every edge.
Alternative: PC algorithm, since independence tests are reliable on large samples.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: noise terms are non-Gaussian; no hidden confounders.

## 24. front_door (en, mode=recommend)

**Question:** Need help: What's the causal effect of remote work policy on productivity? We don't observe manager support, which likely influences who gets remote work policy and also productivity. Remote work policy changes skills test score, and skills test score is what changes productivity; nothing else connects them. Any advice?

```
rows=11145 cols=4 structure=cross_section
col productivity numeric uniq=5531 range=[26.58..173.4]
col skills_score numeric uniq=4585 range=[20.85..119.1]
col remote_work binary uniq=2 values=yes|no
col worker_id integer uniq=10855 range=[1000..1.214e+04] missing=3%
```

**Parse:** `<T:effect> treatment=`remote_work` outcome=`productivity` mediator=`skills_score` hidden=`yes``

```
size rows=11145 covariates=1
overlap auc=0.504 extreme=0 maxsmd=0.018
nonlinearity r2_linear=0.0899 r2_boosting=-0.000992
flags good_overlap large_sample linear_ok
allowed front_door sensitivity_analysis
warning causal_forest diff_in_means dml doubly_robust ipw matching psm regression
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Use Front-door adjustment.
A linear model fits the outcome about as well as boosting (R²=0.0899 vs -0.000992).
Why it fits: front-door works where back-door adjustment is impossible.
Be careful with: Doubly robust estimation (AIPW / DR-learner), because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: the mediator captures the entire effect of the treatment on the outcome; all back-door paths from mediator to outcome are blocked by the treatment.

## 25. instrumental (en, mode=recommend)

**Question:** which method should i use to estimate the effect of `promo` on revenue? but local demand probably affects both `promo` and revenue, and we didn't measure it. distance to the distribution center (`dist_to_dc_km`) influences `promo` and is unrelated to everything else that affects revenue. any advice

```
rows=6519 cols=8 structure=cross_section
col population numeric uniq=6514 range=[0..5.359e+05]
col customer_age numeric uniq=3486 range=[0..81.87]
col sales numeric uniq=6505 range=[257.2..2.062e+04]
col parking_spaces integer uniq=139 range=[0..169]
col competitors integer uniq=13 range=[0..12]
col dist_to_dc_km numeric uniq=6093 range=[0..101.5]
col promo binary uniq=2 values=False|True
col store_age_years numeric uniq=2151 range=[0..35.42]
```

**Parse:** `<T:effect> treatment=`promo` outcome=`sales` instrument=`dist_to_dc_km` hidden=`yes``

```
size rows=6519 covariates=5
overlap auc=0.667 extreme=0 maxsmd=0.472
firststage F=1.55e+03
nonlinearity r2_linear=0.398 r2_boosting=0.292
flags good_overlap imbalance large_sample linear_ok
allowed iv sensitivity_analysis
warning causal_forest dml doubly_robust ipw matching psm regression diff_in_means
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

I recommend Instrumental variables (2SLS).
Use a variable that shifts the treatment but affects the outcome only through it, so the effect is identified even with unobserved confounders.
The instrument is strong: first-stage F=1.55e+03.
A linear model fits the outcome about as well as boosting (R²=0.398 vs 0.292).
Why it fits: IV is one of the few ways to handle unobserved confounders.
Be careful with: Covariate matching, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: exclusion restriction (the instrument affects the outcome only through the treatment); the instrument is as good as random (no confounding of instrument and outcome).

## 26. not_causal (ru, mode=refuse)

**Question:** вопрос: спрогнозируй `cig_sales` на следующий период. как это правильно проанализировать?

```
rows=233 cols=7 structure=cross_section
col retprice numeric uniq=231 range=[13.08..192.9]
col beer numeric uniq=221 range=[11.77..40.16]
col cig_sales numeric uniq=230 range=[0.6..234.6]
col wage_policy binary uniq=2 values=True|False
col population numeric uniq=231 range=[0..9.45e+06]
col unemployment numeric uniq=230 range=[0..11.31]
col county_id integer uniq=233 range=[1000..1232]
```

**Parse:** `<T:not_causal> target=`cig_sales``

```
size rows=233 covariates=7
flags -
allowed -
warning -
```

**Methods:** {}

**Explanation:**

Это вопрос о прогнозе, а не о причинности: важно, насколько хорошо можно предсказать или описать `cig_sales`, а не что будет, если что-то изменить. Используйте предсказательную модель (например, градиентный бустинг или регуляризованную регрессию) с кросс-валидацией или описательную статистику. Причинные методы нужны, только если вы собираетесь вмешиваться и хотите знать эффект.

## 27. panel_factor (en, mode=options)

**Question:** Need help: Some hospitals got treated and otherrs didn't. Did it change readmissionRate? What approach would you recommend?

```
rows=1066 cols=4 structure=panel
panel unit=hospital(41) time=year(26) balanced
col hospital categorical uniq=41 values=hospital_1|hospital_2|hospital_3|hospital_4|hospital_5
col year integer uniq=26 range=[1997..2022]
col readmissionRate numeric uniq=1054 range=[0.2..108.7]
col treated binary uniq=2 range=[0..1] values=0|1
design indicator(treated) treated=2 controls=39 pre=19 post=7 block
```

**Parse:** `<T:effect> treatment=`treated` outcome=`readmissionRate``

```
size rows=1066 covariates=0 treated=2 controls=39 pre=19 post=7
pretrend drift=36.1 rel=2.67 p=1.22e-09
scfit r2=0.997 donors=8 outside=0
flags has_controls policy_design pretrend_violated sc_good_fit
allowed synthetic_control sdid sensitivity_analysis
warning did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.9, 'did': 0.1}

**Explanation:**

There is no single best choice here; Synthetic difference-in-differences and Synthetic control are both good options.
Before the treatment the gap between treated and control units drifted by 36.1 (267% of the outcome's SD, p=1.22e-09), so the trends are not parallel.
A synthetic control reproduces the treated path before the treatment well (R²=0.997).
- Synthetic difference-in-differences: unit and time weights repair non-parallel trends
- Synthetic control: does not require parallel trends, it matches the trajectory instead
Why Synthetic difference-in-differences may be better than Synthetic control: lower variance thanks to fixed effects, and levels need not match.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 28. panel_parallel (en, mode=recommend)

**Question:** my manager asked me this: quantify the effect of `promo` on units sold (unitssold).

```
rows=378 cols=5 structure=panel
panel unit=storeId(54) time=week_start(7) balanced
col storeId categorical uniq=54 values=S001|S002|S003|S004|S005
col promo binary uniq=2 range=[0..1] values=0|1
col parkingSpaces integer uniq=101 range=[0..118] missing=3%
col week_start datetime uniq=7 values=2020-11-30|2021-01-11
col unitsSold numeric uniq=378 range=[22.01..908.4]
design indicator(promo) treated=13 controls=41 pre=4 post=3 block
```

**Parse:** `<T:effect> treatment=`promo` outcome=`unitsSold``

```
size rows=378 covariates=1 treated=13 controls=41 pre=4 post=3
pretrend drift=21.4 rel=0.144 p=0.143
scfit r2=-43.3 donors=41 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_poor_fit short_pre_period
allowed did sensitivity_analysis
warning dml regression sdid synthetic_control
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

I recommend Difference-in-differences.
Compare the change in outcome before and after treatment in the treated group with the same change in the control group.
Pre-treatment trends look parallel (drift 21.4, p=0.143).
Why it fits: pre-treatment trends of the groups are parallel; many treated units make the estimate precise.
Alternative: Synthetic difference-in-differences.
Not recommended: Synthetic control, because the synthetic control does not reproduce the pre-treatment path.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 29. panel_staggered (ru, mode=recommend)

**Question:** `wfhPolicy` -> `turnoverRate`: есть эффект?

```
rows=2825 cols=5 structure=panel
panel unit=employer(113) time=month(25) balanced
col employer categorical uniq=113 values=company_1|company_2|company_3|company_4|company_5
col wfhPolicy binary uniq=2 values=no|yes
col age numeric uniq=1991 range=[0..71.69]
col month datetime uniq=25 values=2013-01-01|2015-01-01
col turnoverRate numeric uniq=1744 range=[0.1..49.4]
design indicator(wfhPolicy) treated=92 controls=21 pre=5 post=20 staggered
```

**Parse:** `<T:effect> treatment=`wfhPolicy` outcome=`turnoverRate``

```
size rows=2825 covariates=1 treated=92 controls=21 pre=5 post=20
pretrend drift=0.44 rel=0.0824 p=0.263
scfit r2=1 donors=21 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit staggered
allowed sdid sensitivity_analysis
warning did synthetic_control dml regression
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

Рекомендуемый метод: Синтетическая разность разностей (SDID).
Взвешенная разность разностей с весами объектов (совпадать должны тренды, а не уровни) и весами периодов.
Тренды до воздействия выглядят параллельными (смещение 0.44, p=0.263).
Почему подходит: воздействие вводили в разное время.
Разность разностей (DiD) здесь хуже: воздействие вводили в разное время.
Проверьте допущения: параллельные тренды после перевзвешивания объектов и периодов; нет перетока эффекта на контрольные объекты.

## 30. rct (en, mode=recommend)

**Question:** I have a CSV with 12589 rows. Free shipping offer -> site visits: is there an effect? How should I anlayse this

```
rows=12589 cols=7 structure=cross_section
col sessions numeric uniq=11546 range=[152.2..1623]
col prev_spend numeric uniq=10864 range=[0..723.1]
col customer_age numeric uniq=4307 range=[0..79.69]
col platform categorical uniq=3 values=mobile|desktop|tablet
col free_shipping binary uniq=2 range=[0..1] values=0|1
col cart_value numeric uniq=11546 range=[0..132.8]
col acquisition_channel categorical uniq=4 values=search|referral|direct|social
```

**Parse:** `<T:effect> treatment=`free_shipping` outcome=`sessions``

```
size rows=12589 covariates=5
overlap auc=0.501 extreme=0 maxsmd=0.0295
nonlinearity r2_linear=0.629 r2_boosting=0.543
flags good_overlap large_sample linear_ok randomization_plausible
allowed regression causal_forest dml doubly_robust ipw matching psm sensitivity_analysis
warning diff_in_means
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

I recommend Difference in means (randomized experiment).
A linear model fits the outcome about as well as boosting (R²=0.629 vs 0.543).
Covariates predict the treatment with AUC=0.501; 0% of units have extreme propensity scores.
Why it fits: covariates are balanced, so randomization looks credible.
Valid only if treatment was randomized; please confirm.
Alternative: Linear regression with an adjustment set, since relationships look linear, so a linear model is adequate.
Assumptions to check: treatment was genuinely randomized; SUTVA (one unit's treatment does not affect another unit's outcome).

## 31. rct_heterogeneous (en, mode=options)

**Question:** Does the remote work policy help some employees more than others when it comes to `attrition`? How should I analyse this?

```
rows=3085 cols=5 structure=cross_section
col company_size integer uniq=914 range=[0..1367]
col age numeric uniq=2092 range=[4.73..70.38]
col last_rating numeric uniq=407 range=[0.63..6.31]
col remote_work binary uniq=2 values=False|True
col attrition numeric uniq=1531 range=[0.1..35.77]
```

**Parse:** `<T:heterogeneity> treatment=`remote_work` outcome=`attrition``

```
size rows=3085 covariates=3
overlap auc=0.485 extreme=0 maxsmd=0.0216
nonlinearity r2_linear=0.167 r2_boosting=0.0162
flags good_overlap linear_ok randomization_plausible
allowed meta_learners causal_forest dml doubly_robust
warning -
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

Several methods fit this data about equally well: Meta-learners (S-, T-, X-learner) and Causal forest (CausalForestDML).
A linear model fits the outcome about as well as boosting (R²=0.167 vs 0.0162).
Covariates predict the treatment with AUC=0.485; 0% of units have extreme propensity scores.
The data has 3085 rows.
- Meta-learners (S-, T-, X-learner): on experimental data meta-learners give an honest estimate of heterogeneity
- Causal forest (CausalForestDML): An ensemble of causal trees that split the sample so that the effect is homogeneous within leaves.
Why Causal forest (CausalForestDML) may be better than Meta-learners (S-, T-, X-learner): provides confidence intervals for CATE.
Assumptions to check: no unobserved confounders (holds automatically in an experiment); overlap between groups on covariates (positivity).

## 32. rdd (ru, mode=recommend)

**Question:** подскажи, помоги понять, влияет ли free_shipping на `конверсия`. как это правильно проанализировать?

```
rows=7035 cols=5 structure=cross_section
col free_shipping binary uniq=2 range=[0..1] values=1|0
col engagement_score numeric uniq=4547 range=[0.4..143.4]
col months_active numeric uniq=3150 range=[0..56.72]
col cart_value numeric uniq=4544 range=[0..124.7]
col конверсия numeric uniq=5129 range=[16.6..291.6]
```

**Parse:** `<T:effect> treatment=`free_shipping` outcome=`конверсия``

```
size rows=7035 covariates=3
overlap auc=1 extreme=0.846 maxsmd=2.61
threshold var=engagement_score cutoff=69.7 acc=0.996 jump=0.972
nonlinearity r2_linear=0.182 r2_boosting=0.172
flags imbalance large_sample linear_ok poor_overlap running_variable_found sharp_cutoff
allowed rdd sensitivity_analysis
warning causal_forest dml doubly_robust regression diff_in_means ipw matching psm
```

**Methods:** {'rdd': 1.0}

**Explanation:**

Лучше всего подойдёт Разрывный регрессионный дизайн (RDD).
Воздействие включается при `engagement_score` >= 69.7: это правило объясняет 100% назначений.
Линейная модель описывает исход почти так же хорошо, как бустинг (R²=0.182 против 0.172).
Ковариаты предсказывают воздействие с AUC=1; у 85% объектов экстремальный propensity score.
Почему подходит: воздействие назначено по порогу непрерывной переменной; порог определяет воздействие почти точно.
Линейная регрессия с контролем конфаундеров здесь хуже: у части объектов вероятность воздействия почти 0 или почти 1.
Проверьте допущения: объекты не могут точно подогнать значение под порог; эффект оценивается только около порога.

## 33. time_series (en, mode=recommend)

**Question:** Quick question: Help me fgure out if the TV advertiisng has any effect on conversion.

```
rows=82 cols=4 structure=time_series
time_series time=period
col period datetime uniq=82 values=2006-05-01|2013-02-01
col Tv On binary uniq=2 values=no|yes
col Conversions numeric uniq=82 range=[93.53..140.2]
col Cart Value numeric uniq=80 range=[12.66..102] missing=2%
```

**Parse:** `<T:effect> treatment=`Tv On` outcome=`Conversions``

```
size rows=82 covariates=1 pre=57 post=25
flags no_controls
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Use Interrupted time series / CausalImpact.
Model the pre-treatment dynamics of the series (optionally with covariate series) and forecast the counterfactual after the intervention.
This is a single series with no control group: 57 periods before the intervention and 25 after.
Why it fits: there is no control group, so a time-series counterfactual is the main option.
Assumptions to check: the pre-treatment model extrapolates correctly into the post-period; covariate series (if used) are not affected by the treatment.

## 34. unidentifiable (ru, mode=refuse)

**Question:** изменение subscriptions вызвано `premium_trial` или это совпадение? боюсь, что неизмеренный фактор (вовлечённость пользователей) влияет и на `premium_trial`, и на subscriptions

```
rows=2014 cols=4 structure=cross_section
col friends integer uniq=42 range=[0..43]
col premium_trial binary uniq=2 range=[0..1] values=0|1
col subscriptions numeric uniq=1691 range=[5.79..128.1]
col device_price numeric uniq=1976 range=[0..1204]
```

**Parse:** `<T:effect> treatment=`premium_trial` outcome=`subscriptions` randomized=`no` hidden=`yes``

```
size rows=2014 covariates=2
overlap auc=0.588 extreme=0 maxsmd=0.32
nonlinearity r2_linear=0.177 r2_boosting=0.0342
flags good_overlap imbalance linear_ok
allowed sensitivity_analysis
warning ipw matching psm regression causal_forest dml doubly_robust
```

**Methods:** {'sensitivity_analysis': 1.0}

**Explanation:**

Эффект `premium_trial` на `subscriptions` нельзя идентифицировать по этим данным. Неизмеренный конфаундер, о котором вы пишете, влияет на оба, а эксперимента, инструмента или медиатора, чтобы разорвать эту связь, нет: регрессия, мэтчинг и DML будут смещены. В данных 2014 строк. Что можно сделать: анализ чувствительности покажет, насколько сильным должен быть скрытый конфаундер, чтобы объяснить связь; лучше провести эксперимент, найти инструмент или измерить конфаундер.

## 35. backdoor_large (en, mode=options)

**Question:** We have data on patients. Which metohd should I use to estimate the effect of the new drug on recovery time? What approach would you recommend?

```
rows=3281 cols=17 structure=cross_section
col survey_q2 numeric uniq=2150 range=[0..7.222]
col cholesterol numeric uniq=3245 range=[89.61..315.3]
col smoker binary uniq=2 values=yes|no
col bmi numeric uniq=2959 range=[13.39..41.5]
col recovery_days numeric uniq=1351 range=[0.08..31.51]
col gender binary uniq=2 values=female|male
col patient_id integer uniq=3281 range=[1000..4280]
col insurance categorical uniq=3 values=public|none|private
col idx1 numeric uniq=2471 range=[0..102.3]
col comorbidities integer uniq=7 range=[0..6]
col severity numeric uniq=848 range=[0..12.1]
col index_4 numeric uniq=2486 range=[0..105.2]
col q5_score numeric uniq=2177 range=[0..7.558]
col patient_age numeric uniq=3185 range=[0..98.4]
col kpi_3 numeric uniq=3216 range=[12.81..187.8]
col treated binary uniq=2 range=[0..1] values=0|1
col idx6 numeric uniq=2462 range=[0..100.7]
```

**Parse:** `<T:effect> treatment=`treated` outcome=`recovery_days``

```
size rows=3281 covariates=15
overlap auc=0.639 extreme=0 maxsmd=0.32
nonlinearity r2_linear=0.17 r2_boosting=0.722
flags good_overlap imbalance nonlinear
allowed causal_forest dml doubly_robust ipw matching psm sensitivity_analysis
warning diff_in_means regression
```

**Methods:** {'dml': 1.0, 'doubly_robust': 1.0, 'causal_forest': 0.6, 'ipw': 0.3, 'psm': 0.3, 'regression': 0.2}

**Explanation:**

Several methods fit this data about equally well: Double machine learning and Doubly robust estimation (AIPW / DR-learner).
The outcome depends on the covariates non-linearly: boosting reaches R²=0.722 vs 0.17 for a linear model.
- Double machine learning: flexible ML models capture the non-linear relationships found in the data
- Doubly robust estimation (AIPW / DR-learner): flexible models in both parts capture the non-linear relationships
Assumptions to check: all confounders are observed; the ML models predict outcome and treatment reasonably well.

## 36. backdoor_small (en, mode=recommend)

**Question:** новый_онбординг -> time in app: is there an effect? users opted into новый_онбординг on their own.

```
rows=384 cols=7 structure=cross_section
col user_id integer uniq=384 range=[1000..1383]
col friends integer uniq=36 range=[0..38]
col новый_онбординг binary uniq=2 range=[0..1] values=0|1
col device_price numeric uniq=381 range=[0..1095]
col session_minutes numeric uniq=359 range=[3.3..48]
col country_tier categorical uniq=3 values=tier1|tier3|tier2
col prior_purchases integer uniq=8 range=[0..7]
```

**Parse:** `<T:effect> treatment=`новый_онбординг` outcome=`session_minutes` randomized=`no``

```
size rows=384 covariates=5
overlap auc=0.73 extreme=0.0026 maxsmd=0.639
nonlinearity r2_linear=0.883 r2_boosting=0.802
flags good_overlap imbalance linear_ok
allowed matching regression sensitivity_analysis
warning ipw psm dml doubly_robust
```

**Methods:** {'regression': 1.0, 'psm': 0.6, 'ipw': 0.5, 'matching': 0.6, 'doubly_robust': 0.4}

**Explanation:**

I recommend Linear regression with an adjustment set.
A linear model fits the outcome about as well as boosting (R²=0.883 vs 0.802).
Why it fits: relationships look linear, so a linear model is adequate.
Alternative: Propensity score matching, since propensity distributions of the groups overlap.
Assumptions to check: all confounders are observed and included; no mediators or colliders are included as controls.

## 37. continuous_treatment (en, mode=recommend)

**Question:** Hello, Would revenue (weekly_sales) have been different without `discount_pct`? Any advice?

```
rows=11267 cols=11 structure=cross_section
col parking_spaces integer uniq=143 range=[0..144]
col region categorical uniq=4 values=south|west|north|east
col n_competitors integer uniq=13 range=[0..12]
col population numeric uniq=11256 range=[0..5.811e+05]
col weekly_sales numeric uniq=11259 range=[2297..2.7e+04]
col discount_pct numeric uniq=8886 range=[0.1..51.74]
col store_age_years numeric uniq=2435 range=[0..37.23]
col staff integer uniq=54 range=[0..55]
col past_purchases integer uniq=23 range=[0..22]
col age numeric uniq=4333 range=[0..91.49] missing=2%
col store_size_sqm numeric uniq=11200 range=[27.13..2324]
```

**Parse:** `<T:effect> treatment=`discount_pct` outcome=`weekly_sales``

```
size rows=11267 covariates=9
nonlinearity r2_linear=0.595 r2_boosting=0.909
flags large_sample nonlinear
allowed causal_forest dml sensitivity_analysis
warning regression
```

**Methods:** {'dml': 1.0, 'causal_forest': 0.6, 'regression': 0.3}

**Explanation:**

I recommend Double machine learning.
The outcome depends on the covariates non-linearly: boosting reaches R²=0.909 vs 0.595 for a linear model.
The data has 11267 rows.
Why it fits: flexible ML models capture the non-linear relationships found in the data; DML shines on large samples.
Alternative: Causal forest (CausalForestDML), since trees capture non-linear relationships automatically.
Not recommended: Linear regression with an adjustment set, because the outcome depends on the covariates non-linearly, so a linear model is misspecified.
Assumptions to check: the ML models predict outcome and treatment reasonably well; overlap (for a continuous treatment, the treatment varies at every covariate value).

## 38. discovery_gaussian (en, mode=options)

**Question:** Need help: Discover the causal structure of city population, `Customer Age`, units sold and `Store Size Sqm`.

```
rows=1972 cols=7 structure=cross_section
col Customer Age numeric uniq=1921 range=[0.24..83.7]
col Store Age Years numeric uniq=1295 range=[0.12..38.54]
col Avg Check numeric uniq=1901 range=[9.936..63.22]
col Store Size Sqm numeric uniq=1956 range=[147.2..2210]
col Quantity numeric uniq=1918 range=[110..985.9]
col Population numeric uniq=1972 range=[1963..5.033e+05]
col N Employees integer uniq=51 range=[0..57]
```

**Parse:** `<T:discovery>`

```
size rows=1972 covariates=7
normality nongaussian=0
flags gaussian_data
allowed ges fci pc
warning notears lingam
```

**Methods:** {'pc': 1.0, 'ges': 1.0, 'notears': 0.6, 'fci': 0.6, 'lingam': 0.1}

**Explanation:**

There is no single best choice here; PC algorithm and Greedy equivalence search are both good options.
0% of the continuous variables are clearly non-Gaussian.
The data has 1972 rows.
- PC algorithm: Constraint-based discovery that removes edges using conditional independence tests and orients edges through v-structures.
- Greedy equivalence search: the default BIC score fits linear Gaussian data
Assumptions to check: faithfulness (independences in data come only from the graph); independent observations.

## 39. discovery_hidden (en, mode=recommend)

**Question:** Hey, Discover the causal structure of parental income (Family Income) and earnings. Some common causes might be missing. What's the right methodology here?

```
rows=542 cols=9 structure=cross_section
col Survey Q1 numeric uniq=284 range=[0.02..6.98]
col Previous Grade numeric uniq=442 range=[0.111..5.117]
col Graduation Rate numeric uniq=533 range=[63.24..114.1]
col Age numeric uniq=338 range=[13.51..18.62]
col Parent Education Years numeric uniq=524 range=[5.63..22.32]
col Salary numeric uniq=542 range=[240..9.562e+04]
col Family Income numeric uniq=542 range=[400..1.523e+05]
col Exam Score numeric uniq=539 range=[17.12..115.6]
col Class Size integer uniq=25 range=[13..38] missing=2%
```

**Parse:** `<T:discovery> hidden=`yes``

```
size rows=542 covariates=9
normality nongaussian=0.667
flags non_gaussian
allowed -
warning fci lingam ges notears pc
```

**Methods:** {'fci': 1.0, 'pc': 0.4, 'ges': 0.3, 'lingam': 0.2}

**Explanation:**

I recommend Fast causal inference (FCI).
An extension of PC that allows hidden confounders and marks edges that may be explained by them.
67% of the continuous variables are clearly non-Gaussian.
Why it fits: designed for data with possible hidden confounders.
Works best with at least 1000 rows, has 542.
Avoid: LiNGAM, because LiNGAM assumes no hidden confounders.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: faithfulness; independent observations.

## 40. discovery_nongaussian (en, mode=recommend)

**Question:** I'm analysing our education data. Learn the causal structure from the data, please. How should I analyse this?

```
rows=3218 cols=8 structure=cross_section
col Parent Education Years numeric uniq=2595 range=[9.489..33.44]
col Graduated Pct numeric uniq=3010 range=[53.91..123]
col Age numeric uniq=678 range=[9.35..21.16]
col Earnings numeric uniq=3216 range=[240..9.593e+04]
col Q1 Score numeric uniq=2144 range=[0.02..7.582]
col Exam Score numeric uniq=2276 range=[0.24..129.5]
col Parent Income numeric uniq=3218 range=[3.952e+04..1.834e+05]
col Previous Grade numeric uniq=1493 range=[0.01..5.989]
```

**Parse:** `<T:discovery>`

```
size rows=3218 covariates=8
normality nongaussian=0.875
flags non_gaussian
allowed lingam fci ges notears pc
warning -
```

**Methods:** {'lingam': 1.0, 'pc': 0.6, 'ges': 0.5, 'notears': 0.5, 'fci': 0.5}

**Explanation:**

I recommend LiNGAM.
Functional discovery that uses non-Gaussian noise in linear relationships to recover the full direction of every edge.
88% of the continuous variables are clearly non-Gaussian.
The data has 3218 rows.
Why it fits: non-Gaussian data lets LiNGAM orient every edge.
Alternative: PC algorithm.
Treat the resulting graph as a hypothesis and check it against domain knowledge.
Assumptions to check: relationships are linear; noise terms are non-Gaussian.

## 41. front_door (en, mode=recommend)

**Question:** Need help: Estimate how push notifications influenced paid_conversions. We don't observe user engagement, which likely influences who gets push notifications and also paid_conversions. The whole effect of push notifications on paid_conversions goes through feature usage. Any advice?

```
rows=11673 cols=5 structure=cross_section
col feature_clicks numeric uniq=2044 range=[0.08..32.02]
col account_age numeric uniq=10159 range=[0..447.1] missing=5%
col paid_conversions numeric uniq=5114 range=[4.58..115.6]
col prior_purchases integer uniq=10 range=[0..9] missing=3%
col got_push binary uniq=2 values=treated|control
```

**Parse:** `<T:effect> treatment=`got_push` outcome=`paid_conversions` mediator=`feature_clicks` hidden=`yes``

```
size rows=11673 covariates=2
overlap auc=0.5 extreme=0 maxsmd=0.0235
nonlinearity r2_linear=0.225 r2_boosting=0.14
flags good_overlap large_sample linear_ok
allowed front_door sensitivity_analysis
warning causal_forest diff_in_means dml doubly_robust ipw matching psm regression
```

**Methods:** {'front_door': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Best fit: Front-door adjustment.
If the whole effect flows through an observed mediator that is not confounded with the outcome, the effect is identified despite hidden treatment-outcome confounding.
A linear model fits the outcome about as well as boosting (R²=0.225 vs 0.14).
Why it fits: front-door works where back-door adjustment is impossible.
Avoid: Doubly robust estimation (AIPW / DR-learner), because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: the mediator captures the entire effect of the treatment on the outcome; no unobserved confounding between treatment and mediator.

## 42. instrumental (en, mode=recommend)

**Question:** i need to know whether the new store layout caused a change in basket size. the distance to the distribution center influences the new store layout and is unrelated to everything else that affects basket size. i worry that unmeasured local demand drives both the new store layout and basket size. thanks!

```
rows=1057 cols=10 structure=cross_section
col competitors integer uniq=12 range=[0..11]
col parking_spaces integer uniq=122 range=[0..143]
col customer_id integer uniq=1057 range=[1000..2056]
col region categorical uniq=4 values=north|east|south|west
col avg_check numeric uniq=889 range=[9.11..61.24]
col new_layout binary uniq=2 values=no|yes
col floor_area numeric uniq=1052 range=[380.7..2110]
col customer_age numeric uniq=1037 range=[2.455..86.81]
col warehouse_distance numeric uniq=963 range=[0..85.81]
col population numeric uniq=1057 range=[0..5.46e+05]
```

**Parse:** `<T:effect> treatment=`new_layout` outcome=`avg_check` instrument=`warehouse_distance` hidden=`yes``

```
size rows=1057 covariates=7
overlap auc=0.693 extreme=0 maxsmd=0.529
firststage F=292.1
nonlinearity r2_linear=0.391 r2_boosting=0.194
flags good_overlap imbalance linear_ok
allowed iv sensitivity_analysis
warning ipw matching psm regression causal_forest diff_in_means dml doubly_robust
```

**Methods:** {'iv': 1.0, 'sensitivity_analysis': 0.4}

**Explanation:**

Use Instrumental variables (2SLS).
Use a variable that shifts the treatment but affects the outcome only through it, so the effect is identified even with unobserved confounders.
The instrument is strong: first-stage F=292.1.
Why it fits: IV is one of the few ways to handle unobserved confounders.
Not recommended: Inverse probability weighting, because it assumes all confounders are observed, but a hidden confounder is suspected.
Assumptions to check: relevance (the instrument really moves the treatment); the instrument is as good as random (no confounding of instrument and outcome).

## 43. not_causal (en, mode=refuse)

**Question:** We have data on students. Train a model for `Salary`.

```
rows=149 cols=5 structure=cross_section
col School Type categorical uniq=3 values=public|charter|private
col Tutoring binary uniq=2 values=False|True
col Age numeric uniq=128 range=[11.27..19.96]
col Absences integer uniq=15 range=[0..14]
col Salary numeric uniq=149 range=[1.642e+04..7.467e+04]
```

**Parse:** `<T:not_causal> target=`Salary``

```
size rows=149 covariates=5
flags small_sample
allowed -
warning -
```

**Methods:** {}

**Explanation:**

This is a prediction question, not a causal one: it asks how well `Salary` can be predicted or described, not what happens if you change something. Use a predictive model (for example gradient boosting or regularised regression) with cross-validation, or descriptive statistics. Causal methods are needed only if you plan to intervene on a variable and want the effect of that change.

## 44. panel_factor (en, mode=options)

**Question:** Quick question: How should I analyse whether the smaller class size affects `testScore`?

```
rows=1368 cols=7 structure=panel
panel unit=schoolId(36) time=yr(38) balanced
col testScore numeric uniq=1358 range=[0.24..174.6]
col post_period binary uniq=2 values=False|True
col yr integer uniq=38 range=[1981..2018]
col daysAbsent integer uniq=16 range=[0..16]
col schoolId categorical uniq=36 values=S001|S002|S003|S004|S005
col schoolType categorical uniq=3 values=public|charter|private
col smallClass binary uniq=2 values=no|yes
design group_post(smallClass,post_period) treated=3 controls=33 pre=31 post=7 block
```

**Parse:** `<T:effect> treatment=`smallClass` outcome=`testScore``

```
size rows=1368 covariates=2 treated=3 controls=33 pre=31 post=7
pretrend drift=46.4 rel=2.15 p=1.56e-18
scfit r2=0.999 donors=11 outside=0
flags has_controls policy_design pretrend_violated sc_good_fit
allowed synthetic_control sdid sensitivity_analysis
warning did dml its regression
```

**Methods:** {'sdid': 1.0, 'synthetic_control': 0.9, 'did': 0.1}

**Explanation:**

Several methods fit this data about equally well: Synthetic difference-in-differences and Synthetic control.
Before the treatment the gap between treated and control units drifted by 46.4 (215% of the outcome's SD, p=1.56e-18), so the trends are not parallel.
A synthetic control reproduces the treated path before the treatment well (R²=0.999).
- Synthetic difference-in-differences: unit and time weights repair non-parallel trends
- Synthetic control: does not require parallel trends, it matches the trajectory instead
Why Synthetic difference-in-differences may be better than Synthetic control: lower variance thanks to fixed effects, and levels need not match.
Assumptions to check: no anticipation of the treatment; SUTVA (no spillovers to control units).

## 45. panel_parallel (ru, mode=recommend)

**Question:** Вопрос: Как `treated` повлиял на systolicBp, ессли сравниить с тем, что было бы без него?

```
rows=3700 cols=5 structure=panel
panel unit=hospitalId(148) time=week_start(25) balanced
col hospitalId integer uniq=148 range=[1..148]
col week_start datetime uniq=25 values=2018-03-05|2018-08-20
col systolicBp numeric uniq=3612 range=[72.39..200.9]
col treated binary uniq=2 values=no|yes
col after_treatment binary uniq=2 values=yes|no
design group_post(treated,after_treatment) treated=68 controls=80 pre=9 post=16 block
```

**Parse:** `<T:effect> treatment=`treated` outcome=`systolicBp``

```
size rows=3700 covariates=0 treated=68 controls=80 pre=9 post=16
pretrend drift=-1.11 rel=0.0525 p=0.366
scfit r2=0.877 donors=8 outside=0
flags has_controls many_treated_units policy_design pretrend_ok
allowed did sdid sensitivity_analysis
warning dml its regression synthetic_control
```

**Methods:** {'did': 1.0, 'sdid': 0.8, 'synthetic_control': 0.2}

**Explanation:**

Лучше всего подойдёт Разность разностей (DiD).
Сравнивает изменение исхода до и после воздействия в группе с воздействием с тем же изменением в контрольной группе.
Тренды до воздействия выглядят параллельными (смещение -1.11, p=0.366).
Синтетический контроль хорошо повторяет траекторию до воздействия (R²=0.877).
Объектов с воздействием 68, контрольных 80; периодов до воздействия 9, после 16.
Почему подходит: тренды до воздействия параллельны; объектов с воздействием много.
Альтернатива: Синтетическая разность разностей (SDID).
Синтетический контроль здесь хуже: объектов с воздействием много.
Проверьте допущения: параллельные тренды без воздействия; нет перетока эффекта на контрольные объекты.

## 46. panel_staggered (en, mode=recommend)

**Question:** My manager asked me this: Does `taxLaw` affect employment rate (employmentRate)? How should I analyse this?

```
rows=780 cols=4 structure=panel
panel unit=country(65) time=year(12) balanced
col country categorical uniq=65 values=S001|S002|S003|S004|S005
col taxLaw binary uniq=2 values=False|True
col employmentRate numeric uniq=674 range=[48.63..88.3]
col year integer uniq=12 range=[1999..2010]
design indicator(taxLaw) treated=51 controls=14 pre=4 post=8 staggered
```

**Parse:** `<T:effect> treatment=`taxLaw` outcome=`employmentRate``

```
size rows=780 covariates=0 treated=51 controls=14 pre=4 post=8
pretrend drift=1.11 rel=0.22 p=0.456
scfit r2=1 donors=6 outside=0
flags has_controls many_treated_units policy_design pretrend_ok sc_good_fit short_pre_period staggered
allowed sensitivity_analysis
warning did sdid dml regression synthetic_control
```

**Methods:** {'sdid': 1.0, 'did': 0.4}

**Explanation:**

I recommend Synthetic difference-in-differences.
Weighted DiD with unit weights (like synthetic control, but only trends must match, not levels) and time weights that focus on pre-periods most similar to the post-period.
Pre-treatment trends look parallel (drift 1.11, p=0.456).
A synthetic control reproduces the treated path before the treatment well (R²=1).
There are 51 treated units and 14 control units, 4 periods before and 8 after the treatment.
Why it fits: staggered adoption is handled by running SDID per adoption cohort and averaging.
Works best with at least 5 pre-treatment periods, has 4.
Be careful with: Difference-in-differences, because naive two-way fixed effects is biased when units are treated at different times.
Assumptions to check: parallel trends after reweighting units and periods; SUTVA (no spillovers to control units).

## 47. rct (en, mode=recommend)

**Question:** How big is the effect of the remote work policy on `wage`?

```
rows=1736 cols=4 structure=cross_section
col education categorical uniq=4 values=phd|bachelor|school|master
col team_size integer uniq=22 range=[0..21]
col wage numeric uniq=1736 range=[1.127e+04..9.604e+04]
col wfh_policy binary uniq=2 range=[0..1] values=0|1
```

**Parse:** `<T:effect> treatment=`wfh_policy` outcome=`wage``

```
size rows=1736 covariates=2
overlap auc=0.491 extreme=0 maxsmd=0.0788
nonlinearity r2_linear=0.171 r2_boosting=0.127
flags good_overlap linear_ok randomization_plausible
allowed regression ipw matching psm doubly_robust sensitivity_analysis
warning diff_in_means causal_forest dml
```

**Methods:** {'diff_in_means': 1.0, 'regression': 0.7}

**Explanation:**

Use Difference in means (randomized experiment).
A linear model fits the outcome about as well as boosting (R²=0.171 vs 0.127).
Why it fits: covariates are balanced, so randomization looks credible.
Valid only if treatment was randomized; please confirm.
Alternative: Linear regression with an adjustment set, since relationships look linear, so a linear model is adequate.
Assumptions to check: treatment was genuinely randomized; no differential attrition between groups.

## 48. rct_heterogeneous (ru, mode=options)

**Question:** Хочу оценить индивидуальный эффект `новая_выкладка` на `avg_check`. Это был рандомизированный A/B-тест. Какой метод выбрать?

```
rows=3755 cols=10 structure=cross_section
col parking_spaces integer uniq=134 range=[0..148]
col past_purchases integer uniq=24 range=[0..23]
col возраст numeric uniq=2380 range=[0..88.84] missing=7%
col avg_check numeric uniq=2175 range=[9.11..63.94]
col competitors integer uniq=12 range=[0..11]
col площадь numeric uniq=3470 range=[0..2216] missing=7%
col новая_выкладка binary uniq=2 range=[0..1] values=0|1
col client_id integer uniq=3755 range=[1000..4754]
col staff integer uniq=54 range=[0..61]
col region categorical uniq=4 values=south|west|north|east
```

**Parse:** `<T:heterogeneity> treatment=`новая_выкладка` outcome=`avg_check` randomized=`yes``

```
size rows=3755 covariates=8
overlap auc=0.509 extreme=0 maxsmd=0.0683
nonlinearity r2_linear=0.521 r2_boosting=0.493
flags good_overlap linear_ok randomization_plausible
allowed meta_learners causal_forest dml doubly_robust
warning -
```

**Methods:** {'meta_learners': 1.0, 'causal_forest': 1.0, 'dml': 0.6, 'doubly_robust': 0.6}

**Explanation:**

Здесь подходят несколько методов примерно одинаково хорошо: Мета-лёрнеры для неоднородного эффекта и Причинный лес.
Линейная модель описывает исход почти так же хорошо, как бустинг (R²=0.521 против 0.493).
Ковариаты предсказывают воздействие с AUC=0.509; у 0% объектов экстремальный propensity score.
В данных 3755 строк.
- Мета-лёрнеры для неоднородного эффекта: ковариаты сбалансированы, рандомизация выглядит правдоподобно
- Причинный лес: Ансамбль причинных деревьев, который находит подгруппы с разным эффектом.
Проверьте допущения: нет ненаблюдаемых конфаундеров (в эксперименте выполняется автоматически); SUTVA.

## 49. rdd (en, mode=recommend)

**Question:** I'm analysing our retail data. Is new store layout worth it in terms of units sold (`units_sold`)? New store layout was given to everyone with loyalty points of at least 427.

```
rows=4013 cols=7 structure=cross_section
col layout_change binary uniq=2 values=True|False
col client_id integer uniq=4013 range=[1000..5012]
col area categorical uniq=4 values=north|east|west|south
col units_sold numeric uniq=3819 range=[97.36..870.9]
col population numeric uniq=4013 range=[5104..5.348e+05]
col staff integer uniq=51 range=[0..51]
col loyalty_points numeric uniq=4002 range=[100.4..899.9]
```

**Parse:** `<T:effect> treatment=`layout_change` outcome=`units_sold` running=`loyalty_points``

```
size rows=4013 covariates=5
overlap auc=1 extreme=0.847 maxsmd=3.41
threshold var=loyalty_points cutoff=426.8 acc=0.999 jump=0.974
nonlinearity r2_linear=0.81 r2_boosting=0.776
flags imbalance linear_ok poor_overlap running_variable_found sharp_cutoff
allowed rdd sensitivity_analysis
warning regression causal_forest diff_in_means dml doubly_robust ipw matching psm
```

**Methods:** {'rdd': 1.0}

**Explanation:**

Recommended method: Regression discontinuity design.
When treatment is assigned by a threshold on a running variable, units just above and just below the threshold are comparable.
Treatment switches at `loyalty_points` >= 426.8: this rule explains 100% of assignments.
A linear model fits the outcome about as well as boosting (R²=0.81 vs 0.776).
Why it fits: treatment jumps at a threshold of a continuous variable; the threshold determines treatment almost perfectly (sharp RDD).
Not recommended: Causal forest (CausalForestDML), because without overlap the ML models extrapolate into regions with no comparable units.
Assumptions to check: units cannot precisely manipulate the running variable around the cutoff; other factors change smoothly at the cutoff.

## 50. time_series (ru, mode=recommend)

**Question:** Привет! Каким методом оценить эффект Redesigned на Quantity? Как это правильно проанализировать?

```
rows=198 cols=5 structure=time_series
time_series time=date
col Competitors integer uniq=9 range=[0..8]
col Quantity numeric uniq=198 range=[369.6..704.4]
col date datetime uniq=198 values=2018-11-23|2019-06-08
col Customer Age numeric uniq=193 range=[23.68..64.32]
col Redesigned binary uniq=2 range=[0..1] values=0|1
```

**Parse:** `<T:effect> treatment=`Redesigned` outcome=`Quantity``

```
size rows=198 covariates=2 pre=159 post=39
flags no_controls
allowed its
warning -
```

**Methods:** {'its': 1.0}

**Explanation:**

Рекомендуемый метод: Прерванные временные ряды / CausalImpact.
Моделирует динамику ряда до вмешательства и прогнозирует контрфакт после него.
Это один ряд без контрольной группы: 159 периодов до вмешательства и 39 после.
Почему подходит: контрольной группы нет.
Проверьте допущения: модель до вмешательства корректно продолжается в период после; одновременно с вмешательством не произошло других событий.
