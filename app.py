"""Causal Method Advisor — веб-интерфейс (Streamlit).

  .venv/Scripts/streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from advisor.knowledge import load_cards
from advisor.pipeline import MODELS_DIR, AdviceError, advise, load_backend
from advisor.profiler import load_csv

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "data" / "demo"
CARDS = load_cards()
MODE_LABEL = {"recommend": ("Recommendation", "green"), "options": ("Several good options", "blue"),
              "refuse": ("Cannot answer causally", "orange")}
STATUS_ICON = {"allowed": "✅", "warning": "⚠️", "forbidden": "⛔"}
TYPE_LABEL = {"effect": "effect of X on Y", "heterogeneity": "for whom does X work",
              "discovery": "what causes what", "not_causal": "prediction / description (not causal)"}

st.set_page_config(page_title="Causal Method Advisor", page_icon="🧭", layout="wide")


BACKENDS = {"baseline": "baseline (classical ML)", "causalgpt": "CausalGPT-mini (our transformer)"}


def neural_available() -> bool:
    """Нейросеть — только локально: на демо в облаке нет PyTorch и весов модели."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return (MODELS_DIR / "causalgpt.pt").exists()


@st.cache_resource
def get_backend(name: str):
    return load_backend(name)


@st.cache_data
def read_csv(source) -> pd.DataFrame:
    return load_csv(source)


def demos() -> list[dict]:
    path = DEMO_DIR / "demos.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


# ---------- боковая панель ----------

with st.sidebar:
    st.header("🧭 Causal Method Advisor")
    st.write("Upload a CSV and ask a causal question. The advisor profiles the data, runs assumption "
             "checks and recommends a causal inference method, or several options if the question is hard.")
    available = ["causalgpt", "baseline"] if neural_available() else ["baseline"]
    backend_name = st.selectbox("Model", available, format_func=BACKENDS.get,
                                help="CausalGPT-mini is a 34M-parameter transformer written and trained from "
                                     "scratch in this project. It needs PyTorch and models/causalgpt.pt, so the "
                                     "online demo runs the classical baseline.")
    st.caption("Sources: Molak, *Causal Inference and Discovery in Python*; "
               "Facure, *Causal Inference for the Brave and True*.")

if not (MODELS_DIR / "baseline.joblib").exists():
    st.error("The baseline model is not trained yet. Run `python scripts/train_baseline.py` first.")
    st.stop()

# ---------- данные и вопрос ----------

st.title("Which causal method fits my question?")
demo_list = demos()
source = st.radio("Data", ["Demo dataset", "Upload CSV"], horizontal=True)
df, default_q = None, ""
if source == "Demo dataset" and demo_list:
    names = [d["file"] for d in demo_list]
    choice = st.selectbox("Demo dataset", names, format_func=lambda f: f"{f} — "
                          + next(d["expected"].split("→")[0].strip() for d in demo_list if d["file"] == f))
    demo = next(d for d in demo_list if d["file"] == choice)
    df, default_q = read_csv(DEMO_DIR / choice), demo["question"]
else:
    up = st.file_uploader("CSV file", type=["csv", "txt"])
    if up is not None:
        try:
            df = read_csv(up)
        except Exception as e:  # кривой CSV — понятное сообщение вместо трейсбэка
            st.error(f"Could not read the file: {e}")

if df is None:
    st.info("Choose a demo dataset or upload a CSV to start.")
    st.stop()

with st.expander(f"Data preview: {df.shape[0]} rows × {df.shape[1]} columns", expanded=False):
    st.dataframe(df.head(20), width="stretch")

question = st.text_area("Your question (English or Russian)", value=default_q, height=90,
                        key=f"q-{source}-{default_q[:20]}")
run = st.button("Advise", type="primary")

state_key = f"{source}|{question}|{df.shape}|{backend_name}"
if run:
    st.session_state.pop("fields_override", None)
    st.session_state["state_key"] = state_key
if st.session_state.get("state_key") != state_key:
    st.stop()

try:
    with st.spinner("Profiling the data and running checks…"):
        advice = advise(df, question, get_backend(backend_name), fields=st.session_state.get("fields_override"))
except AdviceError as e:
    st.error(str(e))
    st.stop()

# ---------- как понят вопрос (можно поправить) ----------

st.subheader("1. How I understood the question")
cols = ["—"] + list(df.columns)
f = advice.fields
with st.form("roles"):
    c1, c2, c3 = st.columns(3)
    qtype = c1.selectbox("Question type", list(TYPE_LABEL), index=list(TYPE_LABEL).index(f.get("type", "effect")),
                         format_func=lambda t: TYPE_LABEL[t])
    role_key = "target" if f.get("type") == "not_causal" else "outcome"
    treatment = c2.selectbox("Treatment (X)", cols, index=cols.index(f["treatment"]) if f.get("treatment") in cols else 0)
    outcome = c3.selectbox("Outcome (Y)", cols, index=cols.index(f[role_key]) if f.get(role_key) in cols else 0)
    c4, c5, c6 = st.columns(3)
    instrument = c4.selectbox("Instrument", cols, index=cols.index(f["instrument"]) if f.get("instrument") in cols else 0)
    mediator = c5.selectbox("Mediator", cols, index=cols.index(f["mediator"]) if f.get("mediator") in cols else 0)
    running = c6.selectbox("Threshold variable", cols, index=cols.index(f["running"]) if f.get("running") in cols else 0)
    c7, c8 = st.columns(2)
    rand_opts = ["unknown", "yes", "no"]
    randomized = c7.radio("Randomized?", rand_opts, horizontal=True, index=rand_opts.index(f.get("randomized", "unknown")))
    hidden = c8.checkbox("A hidden confounder is suspected", value=f.get("hidden") == "yes")
    if st.form_submit_button("Re-run with these roles"):
        new = {"type": qtype}
        pick = {"treatment": treatment, "instrument": instrument, "mediator": mediator, "running": running}
        new.update({k: v for k, v in pick.items() if v != "—"})
        if outcome != "—":
            new["target" if qtype == "not_causal" else "outcome"] = outcome
        if randomized != "unknown":
            new["randomized"] = randomized
        if hidden:
            new["hidden"] = "yes"
        st.session_state["fields_override"] = new
        st.rerun()

# ---------- рекомендация ----------

label, color = MODE_LABEL[advice.mode]
st.subheader("2. Advice")
st.markdown(f":{color}-background[**{label}**]")
left, right = st.columns([3, 2])
with left:
    st.markdown(advice.explanation.replace("\n", "  \n"))
    if advice.explanation_source == "model":
        st.caption("Written by the model; every number in it was checked against the data and diagnostics.")
    for note in advice.notes:
        st.warning(note)
with right:
    if advice.confidence is not None and advice.methods and advice.mode != "refuse":
        top = CARDS[next(iter(advice.methods))].name
        st.metric(f"Confidence in {top}, if the roles in step 1 are right", f"{advice.confidence:.0%}",
                  help="Calibrated on held-out validation questions: of the answers given with about this "
                       "confidence, about this share named the best method. It assumes the question was understood "
                       "as shown in step 1 — check the treatment and outcome there; a wrong role can come with high "
                       "confidence. Below the threshold chosen on validation, or with a close second method, the "
                       "advisor shows options instead of one recommendation.")
    if advice.methods:
        st.caption("Method scores (rules have already removed impossible methods)")
        for m, p in list(advice.methods.items())[:5]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{CARDS[m].name} — {p:.0%}")

# ---------- диагностики ----------

st.subheader("3. Assumption checks")
for r in advice.diagnostics.values():
    if r.applicable:
        flags = " ".join(f"`{x}`" for x in r.flags)
        st.markdown(f"**{r.id.replace('_', ' ')}** — {r.message}  {flags}")
with st.expander("Data profile (what the model sees)"):
    st.code(advice.card.to_text(), language=None)

# ---------- карточки методов и правила ----------

st.subheader("4. Methods")
for m in list(advice.methods)[:3]:
    card = CARDS[m]
    with st.expander(f"{card.name} — estimand: {card.estimand}"):
        st.write(card.summary)
        st.markdown("**Assumptions:**\n" + "\n".join(f"- {a}" for a in card.assumptions))
        st.markdown(f"**Python libraries:** {', '.join(card.libraries)}  \n**Read more:** {'; '.join(card.sources)}")

with st.expander("All methods and the rules that filtered them"):
    rows = [{"": STATUS_ICON[v.status], "method": CARDS[v.method].name, "status": v.status,
             "why": "; ".join(v.reasons + v.cons) or "; ".join(v.pros)} for v in advice.verdicts]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
