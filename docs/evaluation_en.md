# Causal Method Advisor: evaluation report

*Stage 10 of the plan, 16 October 2026. Every number below is produced by scripts in the repository from committed results; the summary table is `eval/results/summary.md`, rebuilt by `python -m eval.summary`.*

## The task

Given a CSV file and a question in plain language ("Did the campaign raise sales?"), recommend the causal inference method that fits the data and the question, explain why, and offer alternatives or refuse when the question is ambiguous or cannot be answered from the data. The model behind it had to be our own, written and trained from scratch: no pretrained weights and no external language models.

## What was built

- **CausalGPT-mini**, a 34M-parameter GPT-style decoder written from scratch in PyTorch (hand-written attention, rotary positions, RMSNorm, SwiGLU, KV cache) with our own byte-level BPE tokenizer of 16,384 tokens.
- **Pretraining** on 23M tokens of open text: English and Russian Wikipedia articles on statistics, econometrics, causal inference and everyday domains, Facure's handbook (MIT licence) and our method cards. 131M tokens seen, 54 minutes on an RTX 3060 Ti.
- **Fine-tuning** on 90k synthetic examples from 18 scenario generators (experiments, back-door with good and poor overlap, IV, front-door, RDD, DiD, synthetic control and SDID, staggered adoption, time series, structure discovery, unidentifiable and non-causal questions) in 7 subject domains. Each scenario is generated from a known causal model, so the right method is known, and the right method recovers the true effect while the naive one is biased.
- **Deterministic statistics around the model.** A profiler finds the data structure (cross-section, panel, time series), diagnostics test the assumptions (parallel pre-trends, overlap and balance, instrument strength, a threshold in a running variable, non-Gaussianity), and a rule engine removes methods the data cannot support. The model never computes statistics: it reads the question, ranks the allowed methods and writes the explanation.
- **Two passes with constrained decoding.** Pass 1 parses the question into a type and roles (treatment, outcome, instrument, mediator, ...), and can only name columns of this CSV. Pass 2 reads the diagnostics and ranks the methods. The method confidence is calibrated on validation data; uncertain answers become options, and an uncertain parse makes the app ask the user to confirm the roles.
- **A classical baseline** with the same pipeline: TF-IDF and logistic regression for the parse, gradient boosting for the ranking. It is the yardstick for the model.

## How it is evaluated

Four question sets, from easiest to hardest:

- **Template test**: 4,986 held-out synthetic questions, split by scenario. It shows whether the model learned the task, but it comes from the same question grammar as training.
- **Hand-written questions**: 93 questions (69 English, 24 Russian) about the 12 demo datasets, written outside the template grammar: synonyms instead of column names, informal text, typos, long context.
- **Traps**: 48 questions where a confident recommendation is wrong: a named hidden confounder with no instrument or mediator, no control group, a prediction question, treatment almost decided by the covariates. The honest answer is a refusal or options.
- **Demo questions**: the 12 questions a visitor of the web app sees first.

The hand-written questions and the traps were written during development, by the same people who built the system. They are a test of phrasings the model never saw, not of users it never met. Questions written by other people are still missing (see the last section).

## Results

The baseline against the model the app ships (`models/causalgpt.pt`, with the calibrated answer mode):

| Set | Metric | Baseline | Model |
|---|---|---|---|
| Template test | Best method, given the correct parse | 99.8% | 99.2% |
| Template test | Best method, end to end | 83.9% | **98.8%** |
| Template test | Treatment and outcome | 85.8% | **99.3%** |
| Template test | Question type, unseen templates | 79.9% | **90.4%** |
| Hand-written | Best method, end to end (80 questions) | 57.5% | **67.5%** |
| Hand-written | Answer mode, end to end | 66.7% | **85.7%** |
| Hand-written | Treatment and outcome | 64.5% | **84.9%** |
| Hand-written | Treatment and outcome, Russian | 54.2% | **91.7%** |
| Hand-written | Question type | **94.6%** | 89.2% |
| Hand-written | Own explanation passes the fact check | — | 98.6% |
| Traps | All, end to end | **58.3%** | 39.6% |
| Traps | All, given the correct parse | 100% | 100% |
| Traps | All, right or the wrong parse caught | — | 95.8% |
| Demo questions | As expected | 11 of 12 | 11 of 12 |

"End to end" means the backend parses the question itself and a wrong treatment or outcome counts as a wrong answer. "Given the correct parse" feeds the gold roles and measures only the ranking and the answer mode.

What the table says:

- **Ranking is solved for both backends.** Given the correct parse, both pick the right method on almost every template question and on all 80 hand-written ones. The difference between them is in reading the question.
- **The model reads questions better.** It links the words of a question to columns ("free delivery" → `free_shipping`, "recovery time" → `recoveryDays`) where the baseline cannot, and its Russian is far better (91.7% against 54.2% of roles). Constrained decoding means it never names a column that does not exist.
- **It is worse at the question type** on hand-written questions: "dose-response effect" is read as a heterogeneity question, "causally related" as a prediction question.
- **It writes its own explanations.** 98.6% pass a check that every number and percentage in the text appears in the model's input and that the text names the recommended method; the rest fall back to a template. Pretraining is what makes this possible: a copy of the model fine-tuned from scratch passes 14.3%, because it fills the same format with numbers that are not in the data.
- **On traps the model is behind end to end** because it misreads the roles on the trap datasets (for example `income` taken as the outcome). With the correct roles both backends answer every trap correctly, and the model's parse check asks the user to confirm almost every wrong parse: right or caught on 46 of 48.

## Is the model better than the baseline?

On the hand-written questions, best method end to end, paired question by question with the baseline (bootstrap interval over questions, exact McNemar test):

| Model run | Model | Difference | 95% interval | p |
|---|---|---|---|---|
| Shipped run (seed 0, current data) | 67.5% | +10.0 | −1.2 to +21.2 | 0.15 |
| Seed 1, current data | 72.5% | +15.0 | +3.8 to +26.2 | 0.02 |
| Seed 1, previous data | 72.5% | +15.0 | +3.8 to +26.2 | 0.02 |
| Stage 8 run (seed 0, earlier data) | 78.8% | +21.3 | +8.8 to +32.5 | 0.002 |

Every run of the recipe is ahead of the baseline, and three of four clearly so. The shipped run on its own is not significantly better: its interval includes zero. Runs of the same recipe differ by 5 to 11 points on these 80 questions, more than the effect of the last data change, so a single run cannot rank recipes and these questions cannot decide between close models. The shipped run was fixed by the protocol (seed 0) before the results were known and was not replaced by the better-scoring run.

## The demo questions

| Dataset | Expected | Baseline | Model |
|---|---|---|---|
| Prop 99 (real data) | SDID / synthetic control | options: SDID | options: SDID, asks to confirm the treatment |
| A/B test | difference in means | recommends it | recommends it (0.98) |
| Training by threshold | RDD | recommends it | recommends it (1.00) |
| Tax and neighbour policy | IV | IV, but reads `urban_share` as the outcome | recommends IV (1.00) |
| One clinic got the drug | SDID / synthetic control | options: SDID | options: SDID (0.91) |
| Staggered tutoring | SDID | recommends it | recommends it (1.00) |
| Push notifications, many covariates | DML / doubly robust | options: DML | options: doubly robust (0.66) |
| App metrics, no hypothesis | LiNGAM | recommends it | recommends it (0.98) |
| Medicaid time series | interrupted time series | recommends it | recommends it (1.00) |
| Push, hidden confounder | refusal + sensitivity analysis | refuses | refuses (1.00) |
| Remote work, full mediator | front-door | recommends it | **IV (1.00), wrong** |
| Physiotherapy hours (Russian) | regression | recommends it | recommends it (0.60) |

On Prop 99 the model reads the roles correctly but its parse confidence is below the threshold, so it asks to confirm the treatment. The reference estimators in the repository reproduce Facure's numbers on this data (DiD −27.35, synthetic control −19.51, SDID −15.61): plain DiD, whose parallel-trends assumption fails here, overstates the effect.

## Where it fails

- **A confident misreading of a role.** On the front-door demo the model reads the mediator `skills_score` ("remote work affects productivity only through the skills score") as an instrument, with parse confidence 0.99999, above the threshold that would ask the user. Once the parse is wrong, the rest of the pipeline is consistent with it: IV is allowed and recommended with confidence 1.00. With the correct parse the model recommends front-door. Neither the calibrated confidence nor the parse check can see an error the model is sure of.
- **Context said in unfamiliar words** ("randomly assigned the flag", "a hard cutoff on appraisal_score") is missed, although the same fields are almost always right on template questions.
- **Run-to-run variation** is larger than the effects we try to measure on 93 questions.
- **The panel training data** still carries synthetic-control fits from before the optimiser changed; rebuilding the panels with the current code changes a synthetic-control flag in about a fifth of them.
- **The online demo runs the baseline**, because the free hosting has no PyTorch; the model runs in the app locally.

## What would improve it next

- **Questions written by other people** — the single most useful thing. They are the only test that is not shaped by the people who wrote the grammar, and more questions would let us tell close models apart.
- **Always confirming a named instrument or mediator.** The data cannot tell them apart: with a hidden confounder both are associated with the treatment and the outcome, and neither the exclusion restriction nor full mediation can be tested. The app should state the assumption in words ("`skills_score` shifts the treatment but affects the outcome only through it — right?") whenever the parse uses one. This would catch the front-door misreading without retraining.
- **An auxiliary loss on the question text**, so the parse generalises to unseen phrasings again (the question type on unseen templates was 92.0% when the loss covered the whole sequence, 82.7–90.4% with the loss on the answer only).
- **Several fine-tuning runs per decision**, or an average of their predictions, instead of one run.

## How to reproduce

```
python -m eval.run_eval --backend causalgpt --ckpt checkpoints/sft_overlap_best.pt --name causalgpt_overlap
python -m eval.handwritten --backend causalgpt --name causalgpt_calibrated --explain
python -m eval.traps --backend causalgpt
python -m eval.demos --backend causalgpt
python -m eval.summary
```

The same commands with `--backend baseline` give the baseline columns. Model checkpoints are not stored in git; the full training recipe is in the README.
