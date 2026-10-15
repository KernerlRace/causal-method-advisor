# Causal Method Advisor

[![tests](https://github.com/KernerlRace/causal-method-advisor/actions/workflows/tests.yml/badge.svg)](https://github.com/KernerlRace/causal-method-advisor/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Give it a CSV and a question in plain language — *"Did the campaign raise sales?"*,
*"What drives churn?"* — and it recommends which causal inference method to use, and why.
If the question is ambiguous or the data cannot answer it, it says so and offers alternatives.

**▶ Live demo: [causal-method-advisor.streamlit.app](https://causal-method-advisor.streamlit.app)** — pick a demo dataset and press *Advise*.
<sub>The free hosting sleeps when idle — the first load can take up to a minute.</sub>

The goal is a small language model written and trained from scratch (no pretrained weights, no Hugging Face)
that understands the question and explains the choice, while deterministic code does the statistics.
The model is trained and wired into the app; the online demo runs the classical-ML baseline, because the free
hosting has no PyTorch — run the app locally to talk to the model.

> **Status:** CausalGPT-mini — our own BPE tokenizer and a 34M-parameter GPT-style decoder, pretrained on Wikipedia
> articles and a causal-inference handbook, then fine-tuned on the task — now answers through the full pipeline with
> constrained decoding. On 93 hand-written questions it picks the right method end to end 79% of the time against 60%
> for the classical baseline, and writes its own explanations, checked number by number against the data.
> Next: calibrated confidence, the "options" mode and honest refusals.

<img src="docs/img/question.png" alt="Asking about California's Proposition 99 on the demo dataset">

On the real Prop 99 data the advisor sees one treated state, 38 controls and pre-treatment trends that are
not parallel (p ≈ 1e-6), so it steers away from plain difference-in-differences:

<img src="docs/img/advice.png" alt="Advice: synthetic DiD or synthetic control, with the assumption checks behind it">

## How it works

```mermaid
flowchart TD
    A[CSV + question] --> B[Profiler<br>cross-section / panel / time series,<br>column roles, treatment design]
    B --> C[Question parser<br>question type, treatment, outcome]
    C --> D[Diagnostics<br>parallel trends, overlap and balance,<br>instrument strength, RDD running variable, ...]
    D --> E[Method ranker + explanation]
    E --> F[Rule engine<br>removes methods the data cannot support]
    F --> G{Answer mode}
    G -->|confident| H[Recommendation: X, because ...]
    G -->|ambiguous| I[Options A / B / C + what is missing]
    G -->|not identifiable| J[Honest refusal + what data to collect]
```

The parser and the ranker are pluggable backends: the classical-ML baseline or the from-scratch model.
Statistics are always computed by code, never by the model — small language models make unpredictable
mistakes in causal reasoning, so the model only reads the question, ranks the methods and explains.

The model runs in two passes. Pass 1 reads the data profile and the question and writes the parse under
**constrained decoding**: every allowed continuation (`treatment=` + each column of *this* CSV, `randomized=yes`, ...,
end of parse) is tokenised into a prefix tree, and each step picks only among the children of the current node, so the
model cannot name a column the data does not have. Pass 2 sees the diagnostics and the rule verdicts; the answer mode
and the method probabilities are read from the softmax over the special `<MODE:*>` and `<M:*>` tokens, rules remove
forbidden methods, and the model writes the explanation. The explanation is shown only if every number and percentage
in it appears in the model's input and it names the top method; otherwise a template explanation built from the same
facts is used, and the UI says so.

## What is inside

| Part | What it does |
|---|---|
| `knowledge/methods/*.yaml` | 22 method cards: requirements, assumptions, arguments for and against |
| `advisor/profiler.py` | CSV structure: cross-section / panel / time series, treatment design in panels |
| `advisor/diagnostics.py` | parallel trends, synthetic control fit, overlap and balance, RDD running variable search, instrument strength, non-Gaussianity |
| `advisor/rules.py` | which methods are allowed, allowed with caveats, or ruled out — and why |
| `advisor/estimators.py` | reference estimators: DiD, SC, SDID, IV, RDD, AIPW, DML, ITS, ... On California Prop 99 they match Facure's results: DiD −27.35, SC −19.51, SDID −15.61 |
| `datagen/scenarios/` | 17 scenario generators (experiments, back-door, IV, front-door, RDD, DiD, SC/SDID, staggered adoption, time series, structure discovery, unidentifiable and non-causal questions) × 7 subject domains, each with a known correct answer |
| `datagen/questions.py`, `explanations.py` | user questions (~120 templates, EN with ~15% RU, typos, conversational style) and target explanations whose numbers come from the diagnostics |
| `advisor/baseline.py` | baseline backend: TF-IDF + logistic regression for question parsing, gradient boosting over (question × column) pairs, one model per method for ranking |
| `advisor/pipeline.py` | end-to-end `advise(df, question, backend)` |
| `app.py` | Streamlit UI: upload a CSV or pick a demo → ask → review the parsed roles → get advice, assumption checks and method cards |
| `model/tokenizer.py` | byte-level BPE written from scratch: 16,384 tokens learned on the open corpus and the task data, 37 of them special (markup, question types, answer modes, one token per method); any string round-trips losslessly, and user text cannot smuggle in special tokens |
| `model/transformer.py` | GPT-style decoder written from scratch: pre-norm blocks with RMSNorm, rotary position embeddings, SwiGLU, hand-written causal attention, tied embeddings, KV cache for generation — 34M parameters with the 16k vocabulary |
| `model/data.py`, `train_pretrain.py`, `sample.py` | texts → token stream on disk → training loop (AdamW, warmup + cosine, bf16, gradient accumulation, resumable checkpoints) → sampling |
| `model/sft.py`, `train_sft.py` | fine-tuning format and loop: loss only on the parse and the answer, a soft label (the gold weights of all suitable methods) at the position of the first method, logits computed only where there is a target |
| `model/decode.py` | two-pass inference with constrained decoding, method probabilities from the softmax, generated explanations with a fact check; `ask` and `export` commands |
| `model/limits.py` | load cap for training and evaluation: short pauses keep the GPU busy at most ~65% of the time, PyTorch threads capped at 75% of the cores |
| `eval/handwritten_questions.jsonl` | 93 questions written by hand outside the template grammar, with gold parses and methods; `eval/handwritten.py` scores any backend on them |
| `datagen/corpus.py` | pretraining corpus: Wikipedia EN/RU articles on statistics, econometrics, causal inference and ML (plus everyday domains), Facure's handbook, our method cards and explanations; resumable polite fetching, cleaning, deduplication, document-level train/val split |
| `data/demo/` | 12 demo datasets with questions, including the real Prop 99 data |

## Results

Baseline on the held-out test set (4,970 questions, split by scenario; `eval/results/baseline_test.json`):

| Metric | Baseline | Rules only |
|---|---|---|
| Best method, given a correct parse (top-1) | 99.9% | 69.1% |
| End-to-end top-1 | 83.6% | — |
| Treatment and outcome found correctly | 85.4% | — |
| Question type / on unseen templates | 98.2% / 78.9% | — |

The baseline's weak spots are synonyms (*salary* ↔ `annual_pay`) and unfamiliar phrasing —
exactly what the language model is meant to fix.

Tokenizer (`models/tokenizer.json`). The first version learned only the task's templates; retrained on the open
corpus plus the task data (~100M characters, 63 s), it reads ordinary text far more efficiently:

| Characters per token | Stage 5: 8k, task data only | Now: 16k, corpus + task data |
|---|---|---|
| Held-out task examples, EN / RU | 4.9 / 5.4 | 4.9 / 5.2 |
| Held-out Wikipedia articles, EN / RU | 2.6 / 2.1 | 4.2 / 3.6 |
| This README / PLAN.md (Russian) | 2.2 / 2.0 | 2.9 / 2.7 |

A full training example is 396 tokens at the median and 829 at most, so it still fits the 1,024-token context.

Language model, first check (`model/transformer.py`, 29.9M parameters with the stage 5 vocabulary). It memorizes a single batch — loss 9.0 → 0.0002
in 300 steps — and after 12 minutes of training on the task texts (1.1 epochs of 35M tokens, 55k tokens/s
on an RTX 3060 Ti in bf16, validation loss 0.42) it writes complete answers in the task format.
Greedy decoding on the same 4,970 test questions (`runs/` is not in git; reproduce with `python -m model.sample checkpoints/sft_lm.pt --n 5000`):

| Metric | Model, 12 min of training | Baseline |
|---|---|---|
| Question type / on unseen templates | 99.3% / 92.0% | 98.2% / 78.9% |
| Treatment and outcome found correctly | 97.2% | 85.4% |
| Whole parse exactly right | 93.7% | 81.6% |
| Best method, given the true parse and diagnostics (top-1) | 99.8% | 99.9% |

This is a sanity check, not the finished advisor. The loss covers the whole sequence rather than only the answer,
there is no constrained decoding yet, and the test questions come from the same generator. Free sampling shows
why constrained decoding is planned: the model writes a question about an `instrument` column that its own data card
does not have. Questions written by people are the real test.

Pretraining (`model/train_pretrain.py`; checkpoints are not in git). The 34M-parameter model trained for 54 minutes
on the open corpus: 4,000 steps of 32,768 tokens (131M tokens, 5.6 epochs of 23M), 44.7k tokens/s, learning rate 1e-3
with cosine decay to 1e-4, dropout 0.1. Validation loss levels off at 2.84 as the learning rate reaches its floor
(the first 200 steps, which start at ln 16,384 ≈ 9.7, are not shown):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/pretrain_loss_dark.png">
  <img src="docs/img/pretrain_loss.png" alt="Train and validation loss of the 34M-parameter model over 4,000 steps; validation flattens at 2.84">
</picture>

| Held-out text | Loss | Perplexity |
|---|---|---|
| English Wikipedia (175 articles) | 3.25 | 25.8 |
| Russian Wikipedia (42 articles) | 3.37 | 28.9 |
| Our explanations | 0.41 | 1.5 |

The model writes fluent, Wikipedia-shaped English with little meaning behind it — the first sample for a neutral prompt:

> **In statistics, the median is** the median with the lowest mean. It can be shown that the median is the mean of the
> tails of the tails, but it is easy to use.

Russian, with six times less text, is weaker and slips into repetition, and domain terms often pull the model into the
style of our explanations, which it has learned best. That is about what 34M parameters and 23M tokens can give; the
question for the next stage is whether this language prior helps on questions written by people, so fine-tuning
will be compared with and without it.

### Fine-tuning on the task (stage 8)

The pretrained model and a randomly initialised copy were fine-tuned identically for one epoch on the 90k training
examples (22 minutes each under the load cap; learning rate 3e-4 from the pretrained weights, 1e-3 from scratch):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/sft_loss_dark.png">
  <img src="docs/img/sft_loss.png" alt="Validation loss on the answer: 0.036 when starting from the pretrained model, 0.062 from scratch">
</picture>

**Template test** (4,970 held-out questions, split by scenario; `eval/results/*_test.json`):

| Metric | Baseline | From scratch | Pretrained |
|---|---|---|---|
| Treatment and outcome found correctly | 85.4% | 98.5% | 98.6% |
| Whole parse exactly right | 81.6% | 96.9% | 97.1% |
| Question type / on unseen templates | 98.2% / 78.9% | 98.4% / 81.5% | 98.5% / 82.7% |
| Best method, given the true parse (top-1) | 99.9% | 99.5% | 99.2% |
| Best method end to end (top-1) | 83.6% | 97.8% | 98.1% |

**Hand-written questions** (93 questions, 69 English and 24 Russian, about the 12 demo datasets;
`eval/results/handwritten_*.json`). They were written outside the template grammar — synonyms instead of column
names (*"free delivery"* for `free_shipping`), informal text and typos, long context, Russian text with English column
names — with the gold answer taken from the scenario each dataset was generated from:

| Metric | Baseline | From scratch | Pretrained |
|---|---|---|---|
| Treatment and outcome found correctly | 65.6% | 83.9% | **86.0%** |
| Best method end to end (top-1, 80 questions) | 60.0% | 72.5% | **78.8%** |
| Answer mode end to end | 69.0% | 79.8% | 83.3% |
| Question type | **94.6%** | 84.9% | 87.1% |
| Context: randomised, hidden confounder, instrument, mediator, threshold | **80.6%** | 75.3% | 78.5% |
| Model's own explanation passes the fact check | — | 14.3% | **98.6%** |

The pretrained model beats the baseline by 18.8 points on the end-to-end method (paired bootstrap 95% CI +7.5 to +30.0;
20 questions only the model gets right, 5 only the baseline). Its gain is in linking the words of a question to a
column; given the correct parse, both pick the right method on all 80 questions. With 80 questions, the 6-point gap
between pretrained and from-scratch is within noise (CI −3.8 to +16.2). Pretraining shows where language matters: the
explanations. Both models learned the format, but the one trained from scratch fills it with numbers that are not in
the data, for example the pre-trend drift of the clinic panel as *108% of the outcome's SD* where the diagnostics say
1.72. With only 5% of the training examples the gap widens: validation loss 0.055 vs 0.211, and 71.6% vs 0%
of explanations pass the check.

An explanation written by the model for *"Only one of our clinics switched to the new drug, the rest kept the old
treatment. ... Did the new drug actually shorten recovery?"* — every number comes from the diagnostics:

> Several methods fit this data about equally well: Synthetic difference-in-differences and Synthetic control.
> Before the treatment the gap between treated and control units drifted by 8.27 (172% of the outcome's SD,
> p=4.17e-07), so the trends are not parallel. A synthetic control reproduces the treated path before the treatment
> well (R²=0.971). There is 1 treated unit and 38 control units, 24 periods before and 8 after the treatment. ...

What does not work yet. Context said differently from the templates — *"randomly assigned the flag"*,
*"a hard cutoff on appraisal_score"* — is missed, although the same fields are almost always right on template
questions; and the question type is worse than the baseline's on these questions (*"dose-response effect"* is read as
heterogeneity, *"causally related"* as a prediction question). On the Prop 99 demo question the pretrained model names
`age15to24` as the treatment, which the baseline gets right: the decoder picks a column token by token, and *"age"*
looked likely. Scoring each column by the probability of its whole name fixes that question but does worse overall
(80 → 78 of 93 roles for the pretrained model, 78 → 72 from scratch), because summed log-probabilities favour short
names, so token-by-token choice stays; the roles can be corrected in the UI. Fine-tuning puts the loss only on the answer, so the
model no longer learns to model the questions themselves: the type on unseen templates fell from 92.0% in the
stage 6 run to 82.7%. These 93 questions were written during development, not by outside users, so the next fixes
(an auxiliary loss on the question, a richer grammar for context) must be measured on questions from other people.

## Quick start

Windows with an NVIDIA GPU, using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe scripts/check_env.py
```

The last command checks the libraries, GPU visibility and runs a short fp16 training step.

Run the web UI with the baseline:

```bash
.venv/Scripts/python.exe -m datagen.build --n 50000 --seed 1 --workers 9
.venv/Scripts/python.exe -m datagen.build_sft --per-scenario 2
.venv/Scripts/python.exe scripts/train_baseline.py --max-parser 40000
.venv/Scripts/streamlit run app.py
```

Other entry points:

```bash
.venv/Scripts/python.exe scripts/demo_stage1.py data/demo/smoking.csv --treatment california --outcome cigsale
.venv/Scripts/python.exe -m eval.run_eval --backend baseline
.venv/Scripts/python.exe -m datagen.corpus titles --lang en      # then --lang ru
.venv/Scripts/python.exe -m datagen.corpus wiki --lang en        # ~50 min; then --lang ru
.venv/Scripts/python.exe -m datagen.corpus facure
.venv/Scripts/python.exe -m datagen.corpus own
.venv/Scripts/python.exe -m model.train_tokenizer --vocab-size 16384 --corpus
.venv/Scripts/python.exe -m model.data corpus --splits train val
.venv/Scripts/python.exe -m model.train_pretrain --steps 4000 --warmup 200 --lr 1e-3 --min-lr 1e-4 --dropout 0.1 --run pretrain
.venv/Scripts/python.exe -m model.sample checkpoints/pretrain_best.pt --loss data/tokens/pretrain_val_wiki_en.bin data/tokens/pretrain_val_wiki_ru.bin data/tokens/pretrain_val_own.bin
.venv/Scripts/python.exe -m model.data sft --splits train val
.venv/Scripts/python.exe -m model.train_pretrain --train data/tokens/sft_train.bin --overfit --steps 300 --batch 8 --accum 1 --lr 1e-3 --min-lr 1e-3 --weight-decay 0 --run overfit
.venv/Scripts/python.exe -m model.train_pretrain --train data/tokens/sft_train.bin --val data/tokens/sft_val.bin --steps 1200 --run sft_lm
.venv/Scripts/python.exe -m model.sample checkpoints/sft_lm.pt --sft test:0
.venv/Scripts/python.exe -m pytest -q tests
```

Fine-tune, evaluate and talk to the model (needs `checkpoints/pretrain_best.pt` from the pretraining step):

```bash
.venv/Scripts/python.exe -m model.sft                       # token cache of the task examples
.venv/Scripts/python.exe -m model.train_sft --init checkpoints/pretrain_best.pt --run sft_pretrained
.venv/Scripts/python.exe -m model.train_sft --init scratch --lr 1e-3 --run sft_scratch
.venv/Scripts/python.exe -m eval.run_eval --backend causalgpt --ckpt checkpoints/sft_pretrained_best.pt --name causalgpt_pretrained
.venv/Scripts/python.exe -m eval.handwritten --backend causalgpt --ckpt checkpoints/sft_pretrained_best.pt --name causalgpt_pretrained --explain
.venv/Scripts/python.exe -m eval.handwritten --backend baseline
.venv/Scripts/python.exe -m model.decode export checkpoints/sft_pretrained_best.pt   # -> models/causalgpt.pt, 65 MB
.venv/Scripts/python.exe -m model.decode ask data/demo/smoking.csv "Did Prop 99 reduce cigarette sales?"
.venv/Scripts/streamlit run app.py                          # the model appears in the sidebar
```

## Roadmap

- [x] Method cards, profiler, diagnostics, rule engine
- [x] Scenario generators with known ground truth, reference estimators
- [x] Question and explanation generators, train/val/test sets
- [x] Classical-ML baseline, end-to-end pipeline, web UI
- [x] Byte-level BPE tokenizer from scratch
- [x] GPT-style decoder from scratch (~30M parameters: attention, RoPE, RMSNorm written by hand)
- [x] Pretraining on an open corpus (Wikipedia EN/RU, Facure's handbook, 23M tokens)
- [x] Fine-tuning on the task, constrained decoding, two-pass inference — beats the baseline on hand-written questions
- [ ] Questions written by other people as the live test set
- [ ] Calibrated confidence, "options" mode and honest refusals
- [ ] Evaluation against the baselines, demo scenarios

The detailed design document (in Russian) is in [PLAN.md](PLAN.md).

## References

- Aleksander Molak, *Causal Inference and Discovery in Python* — used only as a source of knowledge for the method cards,
  never as training text
- Matheus Facure, *Causal Inference for the Brave and True*

## Pretraining data

The pretraining corpus is rebuilt locally by `datagen/corpus.py` and is not stored in this repository:

| Source | Licence | What is used |
|---|---|---|
| [Wikipedia](https://www.wikipedia.org/), English and Russian | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | plain-text articles from statistics, econometrics, causal inference and machine learning categories, plus marketing, health, labour, education and retail articles for everyday vocabulary |
| [Causal Inference for the Brave and True](https://github.com/matheusfacure/python-causality-handbook) by Matheus Facure | MIT | prose of the book's chapters (no code) |
| This project | MIT | method cards and generated explanations from the training split |
