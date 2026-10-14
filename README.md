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
The model exists and trains; the web demo still runs on a classical-ML baseline until the model is wired in.

> **Status:** the non-neural half is done — knowledge base, diagnostics, synthetic data, baseline and web UI.
> The model's own BPE tokenizer and the 30M-parameter GPT-style decoder are written from scratch, and a first
> 12-minute training run already beats the baseline on held-out questions. Next: pretraining and proper fine-tuning.

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

The parser and the ranker are pluggable backends: today a classical-ML baseline, later the from-scratch model.
Statistics are always computed by code, never by the model — small language models make unpredictable
mistakes in causal reasoning, so the model only reads the question and explains.

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
| `model/tokenizer.py` | byte-level BPE written from scratch: 8,192 tokens, 37 of them special (markup, question types, answer modes, one token per method); any string round-trips losslessly, and user text cannot smuggle in special tokens |
| `model/transformer.py` | GPT-style decoder written from scratch: pre-norm blocks with RMSNorm, rotary position embeddings, SwiGLU, hand-written causal attention, tied embeddings, KV cache for generation — 29.9M parameters |
| `model/data.py`, `train_pretrain.py`, `sample.py` | texts → token stream on disk → training loop (AdamW, warmup + cosine, bf16, gradient accumulation, resumable checkpoints) → sampling |
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

Tokenizer (`models/tokenizer.json`, trained on the 90k training examples in ~30 s):

| Metric | Value |
|---|---|
| Characters per token on held-out examples, EN / RU | 4.9 / 5.4 |
| Characters per token on free text it never saw (this README / PLAN.md in Russian) | 2.2 / 2.0 |
| Full training example length, median / p95 / max | 391 / 582 / 827 tokens — fits a 1,024-token context |

The gap between the two rows is expected: the vocabulary has learned the task's templates. It will be retrained
on the pretraining corpus together with the task data before the model is trained.

Language model (`model/transformer.py`, 29.9M parameters). It memorizes a single batch — loss 9.0 → 0.0002
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
.venv/Scripts/python.exe -m model.train_tokenizer --vocab-size 8192
.venv/Scripts/python.exe -m model.data sft --splits train val
.venv/Scripts/python.exe -m model.train_pretrain --overfit --steps 300 --batch 8 --accum 1 --lr 1e-3 --min-lr 1e-3 --weight-decay 0 --run overfit
.venv/Scripts/python.exe -m model.train_pretrain --steps 1200 --run sft_lm
.venv/Scripts/python.exe -m model.sample checkpoints/sft_lm.pt --sft test:0
.venv/Scripts/python.exe -m pytest -q tests
```

## Roadmap

- [x] Method cards, profiler, diagnostics, rule engine
- [x] Scenario generators with known ground truth, reference estimators
- [x] Question and explanation generators, train/val/test sets
- [x] Classical-ML baseline, end-to-end pipeline, web UI
- [x] Byte-level BPE tokenizer from scratch
- [x] GPT-style decoder from scratch (~30M parameters: attention, RoPE, RMSNorm written by hand)
- [ ] Pretraining on an open corpus
- [ ] Fine-tuning on the task, constrained decoding, two-pass inference — must beat the baseline
- [ ] Calibrated confidence, "options" mode and honest refusals
- [ ] Evaluation against the baselines, demo scenarios

The detailed design document (in Russian) is in [PLAN.md](PLAN.md).

## References

- Aleksander Molak, *Causal Inference and Discovery in Python*
- Matheus Facure, *Causal Inference for the Brave and True*
