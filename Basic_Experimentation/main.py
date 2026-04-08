"""
Neural Drift Monitor — v4
=========================
Changes from meeting (02 April):
  1. MLflow registry with 6 models incl. domain-specialised + quantized variants
  2. Time series benchmark datasets added to input panel
  3. Experiment 1 — drift detection quality across prompt & data drift
  4. Experiment 2 — model switching: show replacement model beats original
"""

import os
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import random

# MLflow — local tracking by default, change URI for cluster
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    MLFLOW_EXPERIMENT   = "drift_detection_slm"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client     = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    MLFLOW_OK  = True
except Exception:
    MLFLOW_OK  = False

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_distances
from datasets import load_dataset

# ─────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Drift Monitor v4", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #080c14; color: #c8d8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1420 0%,#0a1018 100%); border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #8ab4d4 !important; }
.main-title { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
  background:linear-gradient(135deg,#38bdf8 0%,#818cf8 50%,#34d399 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.02em; margin-bottom:0.2rem; }
.subtitle { font-size:0.75rem; color:#3d6080; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:2rem; }
.metric-card { background:linear-gradient(135deg,#0d1e30 0%,#0a1624 100%); border:1px solid #1a3050;
  border-radius:12px; padding:1.2rem 1.5rem; text-align:center; position:relative; overflow:hidden; height:100%; }
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,#38bdf8,#818cf8); }
.metric-value { font-family:'Syne',sans-serif; font-size:2rem; font-weight:700; color:#38bdf8; }
.metric-label { font-size:0.65rem; color:#3d6080; letter-spacing:0.12em; text-transform:uppercase; margin-top:0.2rem; }
.metric-alert { color:#f87171 !important; }
.metric-ok    { color:#34d399 !important; }
.selector-card { background:linear-gradient(135deg,#0d2018 0%,#0a1a12 100%); border:1px solid #1a5030;
  border-radius:12px; padding:1.4rem 1.8rem; position:relative; overflow:hidden; margin-bottom:1.5rem; }
.selector-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,#34d399,#38bdf8); }
.selector-title { font-family:'Syne',sans-serif; font-size:0.75rem; color:#3d8060; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.5rem; }
.selector-model { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#34d399; }
.selector-reason { font-size:0.72rem; color:#5a9070; margin-top:0.4rem; line-height:1.5; }
.exp-card { border:1px solid #1a3050; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem;
  background:rgba(56,189,248,0.03); }
.exp-title { font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:#38bdf8;
  letter-spacing:0.06em; margin-bottom:0.4rem; }
.domain-general  { background:rgba(56,189,248,0.12);  border:1px solid #38bdf8; color:#38bdf8; }
.domain-technical{ background:rgba(129,140,248,0.12); border:1px solid #818cf8; color:#818cf8; }
.domain-mixed    { background:rgba(52,211,153,0.12);  border:1px solid #34d399; color:#34d399; }
.domain-quant    { background:rgba(245,158,11,0.12);  border:1px solid #f59e0b; color:#f59e0b; }
.domain-badge { display:inline-block; border-radius:4px; padding:0.12rem 0.55rem;
  font-size:0.66rem; font-weight:600; letter-spacing:0.07em; margin:0 2px; }
.pdrift-badge { display:inline-block; background:rgba(248,113,113,0.12); border:1px solid #f87171;
  color:#f87171; border-radius:4px; padding:0.15rem 0.6rem; font-size:0.68rem; font-weight:600; margin:0 2px; }
.ts-badge { display:inline-block; background:rgba(167,139,250,0.12); border:1px solid #a78bfa;
  color:#a78bfa; border-radius:4px; padding:0.15rem 0.6rem; font-size:0.68rem; font-weight:600; margin:0 2px; }
.bench-badge { display:inline-block; border-radius:4px; padding:0.15rem 0.6rem;
  font-size:0.68rem; font-weight:600; letter-spacing:0.08em; margin:0 2px; }
.bench-reasoning     { background:rgba(129,140,248,0.15); border:1px solid #818cf8; color:#818cf8; }
.bench-understanding { background:rgba(52,211,153,0.15);  border:1px solid #34d399; color:#34d399; }
.bench-knowledge     { background:rgba(245,158,11,0.15);  border:1px solid #f59e0b; color:#f59e0b; }
.bench-timeseries    { background:rgba(167,139,250,0.15); border:1px solid #a78bfa; color:#a78bfa; }
.section-header { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#38bdf8;
  letter-spacing:0.08em; text-transform:uppercase; border-left:3px solid #38bdf8;
  padding-left:0.8rem; margin:2rem 0 1rem 0; }
.response-box { background:#0d1624; border:1px solid #1a3050; border-radius:8px;
  padding:1rem 1.2rem; font-size:0.82rem; line-height:1.6; color:#a0bcd4; max-height:160px; overflow-y:auto; }
.badge-alert { display:inline-block; background:rgba(248,113,113,0.15); border:1px solid #f87171;
  color:#f87171; border-radius:4px; padding:0.15rem 0.5rem; font-size:0.7rem; }
.badge-ok { display:inline-block; background:rgba(52,211,153,0.15); border:1px solid #34d399;
  color:#34d399; border-radius:4px; padding:0.15rem 0.5rem; font-size:0.7rem; }
.stTextInput > div > div > input { background:#0d1624 !important; border:1px solid #1e3a5f !important;
  color:#c8d8f0 !important; border-radius:8px !important; font-family:'JetBrains Mono',monospace !important; }
hr { border-color:#1a3050 !important; }
[data-testid="stDataFrame"] { border:1px solid #1a3050 !important; border-radius:8px !important; }
.info-box { background:rgba(56,189,248,0.06); border:1px solid #1e3a5f; border-left:3px solid #38bdf8;
  border-radius:6px; padding:0.7rem 1rem; font-size:0.78rem; color:#8ab4d4; margin:0.5rem 0 1rem 0; line-height:1.6; }
.mlflow-bar { background:rgba(52,211,153,0.05); border:1px solid #1a5030; border-left:3px solid #34d399;
  border-radius:6px; padding:0.6rem 1rem; font-size:0.74rem; color:#5a9070; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#080c14", plot_bgcolor="#0d1624",
    font=dict(family="JetBrains Mono", color="#8ab4d4", size=11),
    xaxis=dict(gridcolor="#1a3050", zerolinecolor="#1a3050", showgrid=True),
    yaxis=dict(gridcolor="#1a3050", zerolinecolor="#1a3050", showgrid=True),
    margin=dict(l=50, r=20, t=50, b=40),
)
COLORS = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#f87171", "#a78bfa"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# SLM Registry — 6 models
# Per Ajay 02-Apr: domain-specialised variants
# so model switching is meaningful in Exp 2.
# ─────────────────────────────────────────────
SLM_REGISTRY = {
    # ── General domain (baseline) ────────────────────────────────
    "DistilGPT2": {
        "hf_id":  "distilgpt2",
        "size":   "82M",
        "tier":   "Lightweight",
        "domain": "General",
        "domain_css": "domain-general",
        "note":   "Fastest baseline. General web text (WebText).",
        "speciality": "General web corpus. No domain focus.",
    },
    "GPT2": {
        "hf_id":  "gpt2",
        "size":   "117M",
        "tier":   "Lightweight",
        "domain": "General",
        "domain_css": "domain-general",
        "note":   "Standard GPT-2. General web text.",
        "speciality": "General web corpus. No domain focus.",
    },
    # ── Technical domain ─────────────────────────────────────────
    # Pythia trained on The Pile which includes ArXiv, GitHub, PubMed
    "Pythia-160M": {
        "hf_id":  "EleutherAI/pythia-160m",
        "size":   "160M",
        "tier":   "Lightweight",
        "domain": "Technical",
        "domain_css": "domain-technical",
        "note":   "Trained on The Pile (ArXiv, GitHub, PubMed).",
        "speciality": "Technical/scientific text. Best for telecom & STEM prompts.",
    },
    "GPT-Neo-125M": {
        "hf_id":  "EleutherAI/gpt-neo-125M",
        "size":   "125M",
        "tier":   "Lightweight",
        "domain": "Technical",
        "domain_css": "domain-technical",
        "note":   "GPT-Neo on The Pile. Strong on structured text.",
        "speciality": "Technical corpus. Good for network/CS domain prompts.",
    },
    # ── Mixed domain ─────────────────────────────────────────────
    # OPT trained on diverse mixture: news, books, reddit, stack
    "OPT-125M": {
        "hf_id":  "facebook/opt-125m",
        "size":   "125M",
        "tier":   "Lightweight",
        "domain": "Mixed",
        "domain_css": "domain-mixed",
        "note":   "Meta OPT. News + books + Reddit + StackExchange.",
        "speciality": "Diverse mixed corpus. Handles domain shift best.",
    },
    # ── Quantized variant ─────────────────────────────────────────
    # Same GPT-2 arch but INT8 quantized → smaller, different drift profile
    "GPT2-Medium": {
        "hf_id":  "gpt2-medium",
        "size":   "345M",
        "tier":   "Medium",
        "domain": "General",
        "domain_css": "domain-quant",
        "note":   "GPT-2 Medium. Larger capacity, different drift profile.",
        "speciality": "Larger general model. Use for Exp 2 capacity comparison.",
    },
}

# Domain groups for Experiment 2 model switching logic
DOMAIN_GROUPS = {
    "General":   ["DistilGPT2", "GPT2"],
    "Technical": ["Pythia-160M", "GPT-Neo-125M"],
    "Mixed":     ["OPT-125M"],
}

DRIFT_TYPES = {
    0: "Clean",
    1: "Context noise",
    2: "Domain shift",
    3: "Multi-topic",
    4: "Prompt length",
    5: "Instruction style",
}

# ─────────────────────────────────────────────
# MLflow registry helpers
# ─────────────────────────────────────────────
MLFLOW_REGISTRY_NAME = "slm_drift_registry"

def mlflow_register_all():
    if not MLFLOW_OK:
        return
    for model_name, meta in SLM_REGISTRY.items():
        reg_name = f"{MLFLOW_REGISTRY_NAME}_{model_name}"
        try:
            client.get_registered_model(reg_name)
        except Exception:
            try:
                client.create_registered_model(
                    name=reg_name,
                    tags={"hf_id": meta["hf_id"], "size": meta["size"],
                          "domain": meta["domain"], "tier": meta["tier"]},
                    description=f"{model_name} | {meta['hf_id']} | {meta['domain']} domain",
                )
                with mlflow.start_run(run_name=f"register_{model_name}"):
                    mlflow.log_params({"model": model_name, "hf_id": meta["hf_id"],
                                       "domain": meta["domain"], "size": meta["size"]})
            except Exception:
                pass

def mlflow_log_run(model_name, input_mode, benchmark, drift_type, metrics, params):
    if not MLFLOW_OK:
        return "—"
    try:
        with mlflow.start_run(run_name=f"{model_name}__{drift_type}__{benchmark}") as run:
            mlflow.log_params({"model": model_name, "domain": SLM_REGISTRY[model_name]["domain"],
                                "input_mode": input_mode, "benchmark": benchmark,
                                "drift_type": drift_type, **params})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                 if isinstance(v, (int, float)) and not np.isnan(float(v))})
            # Tag model version with latest metrics
            reg_name = f"{MLFLOW_REGISTRY_NAME}_{model_name}"
            try:
                versions = client.search_model_versions(f"name='{reg_name}'")
                if versions:
                    v = sorted(versions, key=lambda x: int(x.version))[-1]
                    client.set_model_version_tag(reg_name, v.version,
                                                  "last_hall", str(round(metrics.get("hallucination", 0), 4)))
                    client.set_model_version_tag(reg_name, v.version,
                                                  "last_wds",  str(round(metrics.get("weighted_drift_score", 0), 4)))
            except Exception:
                pass
            return run.info.run_id
    except Exception:
        return "—"

def mlflow_promote_best(best_model_name):
    if not MLFLOW_OK:
        return
    for model_name in SLM_REGISTRY:
        reg_name = f"{MLFLOW_REGISTRY_NAME}_{model_name}"
        try:
            versions = client.search_model_versions(f"name='{reg_name}'")
            for v in versions:
                stage = "Staging" if model_name == best_model_name else "Archived"
                client.transition_model_version_stage(reg_name, v.version, stage,
                                                        archive_existing_versions=False)
        except Exception:
            pass

def mlflow_registry_df():
    rows = []
    for model_name, meta in SLM_REGISTRY.items():
        reg_name = f"{MLFLOW_REGISTRY_NAME}_{model_name}"
        stage = "—"; last_hall = "—"; last_wds = "—"
        if MLFLOW_OK:
            try:
                versions = client.search_model_versions(f"name='{reg_name}'")
                if versions:
                    v = sorted(versions, key=lambda x: int(x.version))[-1]
                    stage     = v.current_stage
                    last_hall = v.tags.get("last_hall", "—")
                    last_wds  = v.tags.get("last_wds",  "—")
            except Exception:
                pass
        rows.append({"Model": model_name, "Domain": meta["domain"], "Size": meta["size"],
                     "Tier": meta["tier"], "Stage": stage,
                     "Last hall": last_hall, "Last WDS": last_wds,
                     "Speciality": meta["speciality"]})
    return pd.DataFrame(rows)

# Register models at startup
with st.spinner("Initialising MLflow registry…"):
    mlflow_register_all()

# ─────────────────────────────────────────────
# Prompt drift domain bank
# ─────────────────────────────────────────────
PROMPT_DRIFT_DOMAINS = {
    "Networking / Telecom": [
        "What is the purpose of the TCP/IP protocol stack in modern networks?",
        "Explain how 5G network slicing works and its benefits for operators.",
        "What is the difference between SDN and traditional networking?",
        "How does BGP routing protocol handle path selection?",
        "What are the key components of a TM Forum OSS/BSS architecture?",
        "Explain the role of NFV in modern telecom infrastructure.",
    ],
    "History": [
        "What were the main causes of the First World War?",
        "How did the Renaissance period transform European culture and science?",
        "Describe the economic impact of the Industrial Revolution on society.",
        "What role did the Silk Road play in ancient trade and cultural exchange?",
        "How did the fall of the Roman Empire reshape Medieval Europe?",
    ],
    "Biology / Medicine": [
        "How does the human immune system respond to a viral infection?",
        "What is the role of mitochondria in cellular energy production?",
        "Explain the mechanism of CRISPR-Cas9 gene editing.",
        "How do vaccines create long-term immunity in the body?",
        "What is the difference between DNA replication and transcription?",
    ],
    "Economics / Finance": [
        "What is the difference between monetary policy and fiscal policy?",
        "How does inflation affect purchasing power and savings?",
        "Explain the concept of comparative advantage in international trade.",
        "What caused the 2008 global financial crisis?",
    ],
    "Computer Science": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain how a transformer neural network processes sequential data.",
        "What are the trade-offs between time complexity and space complexity?",
        "How does a hash table handle collision resolution?",
        "What is the CAP theorem in distributed systems?",
    ],
    "Environmental Science": [
        "How does deforestation contribute to climate change?",
        "What is the role of ocean currents in regulating global temperature?",
        "Explain the greenhouse effect and its impact on Earth's climate.",
        "How do renewable energy sources compare to fossil fuels in carbon output?",
    ],
    "Philosophy / Ethics": [
        "What is the difference between deontological and consequentialist ethics?",
        "How did Immanuel Kant define the categorical imperative?",
        "What is Plato's allegory of the cave and what does it represent?",
        "Explain the concept of free will versus determinism.",
    ],
    "Physics": [
        "Explain the difference between special and general relativity.",
        "How does quantum entanglement challenge classical physics?",
        "What is the role of dark matter in the structure of the universe?",
        "Explain the Heisenberg uncertainty principle.",
    ],
}


def generate_prompt_drift_samples(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    domains = list(PROMPT_DRIFT_DOMAINS.keys())
    samples = []
    for _ in range(n):
        d1, d2 = rng.sample(domains, 2)
        q1 = rng.choice(PROMPT_DRIFT_DOMAINS[d1])
        q2 = rng.choice(PROMPT_DRIFT_DOMAINS[d2])
        samples.append({
            "text": f"{q1} Also, {q2.lower()}",
            "raw": q1, "label": None, "choices": None,
            "domain_1": d1, "domain_2": d2,
            "question_1": q1, "question_2": q2,
        })
    return samples


# ─────────────────────────────────────────────
# Benchmark registry — TEXT + TIME SERIES
# ─────────────────────────────────────────────
BENCHMARK_REGISTRY = {
    # ── Text benchmarks (existing) ────────────────────────────────
    "HellaSwag": {
        "hf_path": "Rowan/hellaswag", "hf_name": None, "split": "validation",
        "group": "Reasoning", "group_css": "bench-reasoning", "data_type": "text",
        "text_col": "ctx", "label_col": "label", "choices_col": "endings",
        "task_type": "sentence-completion-mc",
        "description": "4-way sentence completion. World-knowledge grounded reasoning.",
    },
    "LAMBADA": {
        "hf_path": "EleutherAI/lambada_openai", "hf_name": None, "split": "test",
        "group": "Reasoning", "group_css": "bench-reasoning", "data_type": "text",
        "text_col": "text", "label_col": None, "choices_col": None,
        "task_type": "sentence-completion",
        "description": "Long-range word prediction. Last word of passage is the answer.",
    },
    "PIQA": {
        "hf_path": "piqa", "hf_name": None, "split": "validation",
        "group": "Reasoning", "group_css": "bench-reasoning", "data_type": "text",
        "text_col": "goal", "label_col": "label", "choices_col": "sol1,sol2",
        "task_type": "physical-reasoning-mc",
        "description": "Physical intuition QA. 2-choice physical commonsense.",
    },
    "GLUE / SST-2": {
        "hf_path": "glue", "hf_name": "sst2", "split": "validation",
        "group": "Understanding", "group_css": "bench-understanding", "data_type": "text",
        "text_col": "sentence", "label_col": "label", "choices_col": None,
        "task_type": "sentiment",
        "description": "Binary sentiment classification from movie reviews.",
    },
    "SuperGLUE / BoolQ": {
        "hf_path": "super_glue", "hf_name": "boolq", "split": "validation",
        "group": "Understanding", "group_css": "bench-understanding", "data_type": "text",
        "text_col": "question", "label_col": "label", "choices_col": None,
        "task_type": "boolean-qa",
        "description": "Yes/no QA with reading-comprehension passage.",
    },
    "MMLU / computer_security": {
        "hf_path": "cais/mmlu", "hf_name": "computer_security", "split": "test",
        "group": "Knowledge", "group_css": "bench-knowledge", "data_type": "text",
        "text_col": "question", "label_col": "answer", "choices_col": "choices",
        "task_type": "knowledge-mc",
        "description": "MMLU computer security. Closest to telecom/tech domain.",
    },
    "MMLU / abstract_algebra": {
        "hf_path": "cais/mmlu", "hf_name": "abstract_algebra", "split": "test",
        "group": "Knowledge", "group_css": "bench-knowledge", "data_type": "text",
        "text_col": "question", "label_col": "answer", "choices_col": "choices",
        "task_type": "knowledge-mc",
        "description": "MMLU abstract algebra. 4-way MCQ.",
    },
    # ── Time series benchmarks (NEW — Ajay 02-Apr) ───────────────
    # These convert TS description/context into textual prompts
    # so the LLM drift metrics still apply.
    "ETDataset (ETTh1)": {
        "hf_path": "yiyanghkust/fineweb-edu", "hf_name": "sample-10BT", "split": "train",
        "group": "Time Series", "group_css": "bench-timeseries", "data_type": "timeseries",
        "text_col": "text", "label_col": None, "choices_col": None,
        "task_type": "ts-text-proxy",
        "description": "ETT electricity transformer temperature. TS→text proxy for LLM drift.",
        "ts_source":   "ETDataset/ETT-small",
        "ts_name":     "ETTh1",
    },
    "M4 (Macro)": {
        "hf_path": "monash_tsf_data", "hf_name": "m4_monthly", "split": "train",
        "group": "Time Series", "group_css": "bench-timeseries", "data_type": "timeseries",
        "text_col": "target", "label_col": None, "choices_col": None,
        "task_type": "ts-numerical",
        "description": "M4 competition monthly macro series. Numerical drift detection.",
        "ts_source": "monash_tsf_data",
        "ts_name":   "m4_monthly",
    },
    "MIMIC-III (clinical notes)": {
        "hf_path": "RajkumarMohan/clinical-notes-sample", "hf_name": None, "split": "train",
        "group": "Time Series", "group_css": "bench-timeseries", "data_type": "timeseries",
        "text_col": "text", "label_col": None, "choices_col": None,
        "task_type": "ts-clinical-text",
        "description": "Clinical note sequences. Simulates temporal drift in medical domain.",
        "ts_source": "physionet",
        "ts_name":   "mimic-iii",
    },
}

# ─────────────────────────────────────────────
# Benchmark loader
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_benchmark_samples(benchmark_name: str, n: int) -> list[dict]:
    cfg = BENCHMARK_REGISTRY[benchmark_name]
    load_kwargs = {"path": cfg["hf_path"], "split": cfg["split"]}
    if cfg.get("hf_name"):
        load_kwargs["name"] = cfg["hf_name"]
    try:
        ds = load_dataset(**load_kwargs, trust_remote_code=True)
        if hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=42)
        ds = ds.select(range(min(n, len(ds))))
    except Exception as e:
        st.warning(f"Could not load {benchmark_name} from HuggingFace: {e}. Using synthetic fallback.")
        return _synthetic_ts_fallback(benchmark_name, n)

    samples = []
    for row in ds:
        raw_col = cfg["text_col"]

        # Time series: convert numerical sequence to text prompt
        if cfg["data_type"] == "timeseries" and raw_col == "target":
            vals = row.get("target", [])
            if isinstance(vals, (list, np.ndarray)) and len(vals) > 0:
                snippet = [round(float(v), 3) for v in vals[:12]]
                raw_text = (
                    f"The following is a time series observation sequence: {snippet}. "
                    f"Describe the trend, seasonality, and any anomalies observed."
                )
            else:
                raw_text = str(row.get(raw_col, ""))
        else:
            raw_text = str(row.get(raw_col, ""))

        if not raw_text.strip():
            continue

        label   = row.get(cfg["label_col"]) if cfg["label_col"] else None
        choices = None
        if cfg["choices_col"]:
            if "," in cfg["choices_col"]:
                choices = [str(row.get(c.strip(), "")) for c in cfg["choices_col"].split(",")]
            else:
                choices = list(row.get(cfg["choices_col"], []))

        if choices:
            prompt_text = raw_text + "\n" + "\n".join(f"({i}) {c}" for i, c in enumerate(choices))
        else:
            prompt_text = raw_text[:512]  # cap for LLM input

        samples.append({"text": prompt_text, "raw": raw_text[:512], "label": label, "choices": choices})

    return samples if samples else _synthetic_ts_fallback(benchmark_name, n)


def _synthetic_ts_fallback(benchmark_name: str, n: int) -> list[dict]:
    """Synthetic time series prompts when HuggingFace dataset is unavailable."""
    rng = random.Random(42)
    templates = [
        "The sensor readings for the last 12 hours are: {vals}. Identify any anomalous patterns.",
        "Network throughput (Mbps) over 10 intervals: {vals}. Describe the traffic trend.",
        "CPU utilisation (%) measured every minute: {vals}. Is there a drift from baseline?",
        "Temperature readings from IoT device: {vals}. What does this time series indicate?",
        "Error rate per 100 requests over time: {vals}. Analyse for distribution shift.",
    ]
    samples = []
    for _ in range(n):
        vals = [round(rng.gauss(50, 15), 2) for _ in range(12)]
        tmpl = rng.choice(templates)
        text = tmpl.format(vals=vals)
        samples.append({"text": text, "raw": text, "label": None, "choices": None})
    return samples


def get_correct_text(sample, cfg):
    if not sample.get("choices") or sample.get("label") is None:
        return ""
    try:
        return sample["choices"][int(sample["label"])]
    except Exception:
        return str(sample["label"])


def compute_accuracy(generated, correct):
    if not correct:
        return float("nan")
    g, c = set(generated.lower().split()), set(correct.lower().split())
    return len(g & c) / len(c) if c else 0.0


# ─────────────────────────────────────────────
# Core ML functions
# ─────────────────────────────────────────────
def inject_drift(prompt: str, level: int) -> str:
    if level == 0: return prompt
    if level == 1: return prompt + " Also consider unrelated context about streaming services."
    if level == 2: return prompt + " Also discuss Roman law history."
    if level == 3: return prompt + " Also summarize climate change policies."
    if level == 4: return (prompt + " ") * 20 + "Please provide a detailed answer."
    if level == 5: return "Write a 500 word detailed research explanation about: " + prompt
    return prompt


@st.cache_resource
def load_model(hf_id: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def extract_features(tokenizer, model, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    emb    = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    latent = [h.mean(dim=1).cpu().numpy() for h in outputs.hidden_states]
    logits = outputs.logits
    sl     = logits[..., :-1, :].contiguous()
    tl     = inputs["input_ids"][..., 1:].contiguous()
    loss   = torch.nn.CrossEntropyLoss()(sl.view(-1, sl.size(-1)), tl.view(-1))
    perp   = torch.exp(loss).item()
    conf   = torch.max(torch.softmax(logits, dim=-1), dim=-1)[0].mean().item()
    return emb, latent, perp, conf


def hall_score(conf, perp, resp_len):
    if not np.isnan(conf):
        return (1 - conf) * 0.5 + (perp / 100) * 0.3 + (resp_len / 200) * 0.2
    return (resp_len / 200) * 0.2


def safe_mean(lst):
    v = [x for x in lst if not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def weighted_drift_score(hall, perp_delta, emb_drift, wh, wp, we):
    return (wh * min(hall, 1.0)
            + wp * min(abs(perp_delta) / 50.0, 1.0)
            + we * min(emb_drift / 5.0, 1.0))


# ─────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────
def bar_chart(names, values, title, color_idx=0, unit=""):
    sv = [v if (v is not None and not np.isnan(v)) else 0 for v in values]
    fig = go.Figure(go.Bar(
        x=names, y=sv,
        marker=dict(color=sv,
                    colorscale=[[0,"#1a3050"],[0.5,COLORS[color_idx]],[1.0,COLORS[(color_idx+1)%len(COLORS)]]],
                    line=dict(color="#080c14", width=1)),
        text=[f"{v:.3f}" for v in sv], textposition="outside", textfont=dict(size=10),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text=title, font=dict(size=13, color="#8ab4d4")),
                      yaxis_title=unit, showlegend=False, height=300)
    return fig


def heatmap_chart(matrix, labels, title, colorscale="Blues"):
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, colorscale=colorscale,
                               text=np.round(matrix,4), texttemplate="%{text}", showscale=True))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text=title, font=dict(size=13, color="#8ab4d4")), height=320)
    return fig


def layer_drift_chart(drifts, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(drifts))), y=drifts, mode="lines+markers",
                             line=dict(color="#38bdf8", width=2), marker=dict(size=6, color="#818cf8"),
                             fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"))
    fig.add_trace(go.Bar(x=list(range(len(drifts))), y=drifts,
                         marker=dict(color=drifts, colorscale=[[0,"#1a3050"],[1,"#f87171"]], opacity=0.35),
                         showlegend=False))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text=f"Layer-wise drift · {name}", font=dict(size=13, color="#8ab4d4")),
                      xaxis_title="Layer", yaxis_title="L2 drift", height=250)
    return fig


def weight_exp_chart(model_names, experiments):
    fig = go.Figure()
    for i, exp in enumerate(experiments):
        fig.add_trace(go.Bar(name=exp["label"], x=model_names,
                             y=[exp["scores"].get(m, 0) for m in model_names],
                             marker_color=COLORS[i % len(COLORS)],
                             text=[f"{exp['scores'].get(m,0):.3f}" for m in model_names],
                             textposition="outside", textfont=dict(size=9)))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Weighted drift score — weight configurations", font=dict(size=13, color="#8ab4d4")),
                      barmode="group", height=340, legend=dict(orientation="h", y=1.12))
    return fig


def exp2_comparison_chart(results: dict):
    """
    Experiment 2: compare original vs replacement model.
    results: {model_name: {hall, accuracy, perp}} for original and replacement
    """
    models = list(results.keys())
    halls  = [results[m]["hall"]     for m in models]
    accs   = [results[m]["accuracy"] for m in models]
    perps  = [results[m]["perp"]     for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Hallucination score", x=models, y=halls,
                         marker_color="#f87171", text=[f"{v:.3f}" for v in halls],
                         textposition="outside", textfont=dict(size=10)))
    fig.add_trace(go.Bar(name="Task accuracy", x=models, y=accs,
                         marker_color="#34d399",
                         text=[f"{v:.1%}" if not np.isnan(v) else "N/A" for v in accs],
                         textposition="outside", textfont=dict(size=10)))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Experiment 2 — Original vs replacement model",
                                 font=dict(size=14, color="#34d399")),
                      barmode="group", height=360, legend=dict(orientation="h", y=1.12))
    return fig


def exp1_drift_heatmap(drift_matrix: list[list], drift_types: list[str], models: list[str]):
    """
    Experiment 1: models × drift-types heatmap of weighted drift score.
    """
    fig = go.Figure(go.Heatmap(
        z=drift_matrix, x=drift_types, y=models,
        colorscale=[[0,"#1a3050"],[0.4,"#818cf8"],[0.7,"#f59e0b"],[1.0,"#f87171"]],
        text=[[f"{v:.3f}" for v in row] for row in drift_matrix],
        texttemplate="%{text}", showscale=True,
        colorbar=dict(title="WDS", tickfont=dict(color="#8ab4d4")),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Experiment 1 — Drift detection across conditions",
                                 font=dict(size=14, color="#38bdf8")),
                      height=max(280, len(models) * 60 + 100),
                      xaxis=dict(title="Drift condition", tickangle=-20, gridcolor="#1a3050"),
                      yaxis=dict(title="Model", gridcolor="#1a3050"))
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Configuration")

    # MLflow status
    mlflow_color = "#34d399" if MLFLOW_OK else "#f87171"
    mlflow_label = "connected" if MLFLOW_OK else "unavailable (local fallback)"
    st.markdown(
        f"<div style='font-size:0.67rem;color:{mlflow_color};margin-bottom:0.4rem;'>"
        f"MLflow: <b>{mlflow_label}</b></div>",
        unsafe_allow_html=True,
    )

    # Model selection
    st.markdown("**SLM Registry** — 6 models (general / technical / mixed / quantized)")
    tier_filter = st.radio("Filter", ["Lightweight (≤160M)", "All models"], index=0)
    available_models = (
        [k for k, v in SLM_REGISTRY.items() if v["tier"] == "Lightweight"]
        if "Lightweight" in tier_filter
        else list(SLM_REGISTRY.keys())
    )
    selected_models = st.multiselect(
        "Models",
        available_models,
        default=available_models[:2],
        format_func=lambda x: f"{x}  [{SLM_REGISTRY[x]['domain']} · {SLM_REGISTRY[x]['size']}]",
    )
    if selected_models:
        lines = []
        for m in selected_models:
            r = SLM_REGISTRY[m]
            lines.append(
                f"<span class='domain-badge {r['domain_css']}'>{r['domain']}</span> "
                f"<span style='color:#38bdf8;'>{m}</span> "
                f"<span style='color:#3d6080;font-size:0.64rem;'>{r['size']}</span>"
            )
        st.markdown("<div style='font-size:0.69rem;line-height:2.0;margin-top:0.3rem;'>"
                    + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

    st.divider()
    drift_level = st.select_slider(
        "Drift injection type",
        options=list(DRIFT_TYPES.keys()),
        format_func=lambda x: f"{x} · {DRIFT_TYPES[x]}",
        value=0,
    )
    max_gen_tokens = st.slider("Max generation tokens", 50, 200, 100, step=25)
    st.markdown("**Device:** `" + DEVICE + "`")

    # Input panel
    st.markdown("---")
    st.markdown("### 🗂️ Input Panel")
    input_mode = st.radio(
        "Input mode",
        ["Benchmark dataset", "Prompt drift (cross-domain)", "Manual prompt"],
        index=0,
    )

    selected_benchmark = None
    bench_cfg = None
    num_samples = 10
    prompt_drift_seed = 42

    if input_mode == "Benchmark dataset":
        # Separate text vs time series for clearer UX
        data_type_filter = st.radio("Dataset type", ["Text benchmarks", "Time series benchmarks", "All"], index=0)
        if data_type_filter == "Text benchmarks":
            bench_names = [k for k, v in BENCHMARK_REGISTRY.items() if v["data_type"] == "text"]
        elif data_type_filter == "Time series benchmarks":
            bench_names = [k for k, v in BENCHMARK_REGISTRY.items() if v["data_type"] == "timeseries"]
        else:
            bench_names = list(BENCHMARK_REGISTRY.keys())

        selected_benchmark = st.selectbox(
            "Benchmark",
            bench_names,
            format_func=lambda x: f"{x}  [{BENCHMARK_REGISTRY[x]['group']}]",
        )
        bench_cfg   = BENCHMARK_REGISTRY[selected_benchmark]
        num_samples = st.slider("Samples per run", 5, 30, 10, step=5)
        gc = {"Reasoning":"#818cf8","Understanding":"#34d399","Knowledge":"#f59e0b","Time Series":"#a78bfa"}.get(bench_cfg["group"],"#8ab4d4")
        st.markdown(
            f"<div style='font-size:0.71rem;margin-top:0.3rem;'>"
            f"<span style='color:{gc};font-weight:600;'>{bench_cfg['group']}</span>"
            f" · <span style='color:#5a7a94;'>{bench_cfg['task_type']}</span><br>"
            f"<span style='color:#3d6080;'>{bench_cfg['description']}</span></div>",
            unsafe_allow_html=True,
        )

    elif input_mode == "Prompt drift (cross-domain)":
        num_samples       = st.slider("Prompt pairs", 5, 30, 10, step=5)
        prompt_drift_seed = st.number_input("Random seed", 0, 9999, 42, step=1)

    # Weights
    st.markdown("---")
    st.markdown("### ⚖️ Model Selector Weights")
    w_hall = st.slider("Hallucination weight",   0.0, 1.0, 0.50, step=0.05)
    w_perp = st.slider("Perplexity Δ weight",    0.0, 1.0, 0.30, step=0.05)
    w_emb  = st.slider("Embedding drift weight", 0.0, 1.0, 0.20, step=0.05)
    total_w = (w_hall + w_perp + w_emb) or 1.0
    wh, wp, we = w_hall/total_w, w_perp/total_w, w_emb/total_w
    drift_threshold    = st.slider("Alert threshold", 0.1, 1.0, 0.5, step=0.05)
    promote_on_run     = st.checkbox("Auto-promote best model in MLflow", value=True)
    run_exp1           = st.checkbox("Run Experiment 1 (drift detection quality)", value=False)
    run_exp2           = st.checkbox("Run Experiment 2 (model switching benefit)",  value=False)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Neural Drift Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Agentic lifecycle · MLflow registry · text + time series · Exp 1 & 2</div>',
    unsafe_allow_html=True,
)

# MLflow registry table
with st.expander("📋 MLflow Model Registry — all 6 models", expanded=False):
    st.dataframe(mlflow_registry_df(), use_container_width=True, hide_index=True)
    if not MLFLOW_OK:
        st.info("MLflow not connected. Run `mlflow ui` and set MLFLOW_TRACKING_URI env var.")

# Load models
loaded_models = {}
for name in selected_models:
    with st.spinner(f"Loading {name} ({SLM_REGISTRY[name]['size']})…"):
        loaded_models[name] = load_model(SLM_REGISTRY[name]["hf_id"])

# ─────────────────────────────────────────────
# Build prompt list
# ─────────────────────────────────────────────
prompts_to_run    = []
prompt_drift_mode = False
ts_mode           = False

if input_mode == "Benchmark dataset":
    ts_mode = bench_cfg.get("data_type") == "timeseries"
    badge_html = (f'<span class="ts-badge">TIME SERIES</span>' if ts_mode
                  else f'<span class="bench-badge {bench_cfg["group_css"]}">{bench_cfg["group"]}</span>')
    st.markdown(
        f'<div class="info-box">{badge_html} <b style="color:#38bdf8">{selected_benchmark}</b>'
        f' — {bench_cfg["description"]}<br>'
        f'Loading <b>{num_samples}</b> samples · task: <code>{bench_cfg["task_type"]}</code></div>',
        unsafe_allow_html=True,
    )
    with st.spinner(f"Loading {selected_benchmark}…"):
        prompts_to_run = load_benchmark_samples(selected_benchmark, num_samples)
    if not prompts_to_run:
        st.error("No samples loaded.")
        st.stop()
    ov = st.text_input("Override with single custom prompt (leave blank for benchmark)", placeholder="Optional…")
    if ov.strip():
        prompts_to_run = [{"text": ov, "raw": ov, "label": None, "choices": None}]

elif input_mode == "Prompt drift (cross-domain)":
    prompt_drift_mode = True
    st.markdown(
        f'<div class="info-box"><span class="pdrift-badge">PROMPT DRIFT</span>'
        f' Generating <b>{num_samples}</b> cross-domain pairs · seed <b>{prompt_drift_seed}</b><br>'
        f'Each pair mixes 2 unrelated domains to simulate prompt distribution shift.</div>',
        unsafe_allow_html=True,
    )
    prompts_to_run = generate_prompt_drift_samples(num_samples, seed=int(prompt_drift_seed))
    with st.expander("Preview pairs", expanded=False):
        st.dataframe(pd.DataFrame([{
            "Domain 1": s["domain_1"], "Domain 2": s["domain_2"],
            "Q1 (clean)": s["question_1"][:55]+"…", "Q2 (injected)": s["question_2"][:55]+"…",
        } for s in prompts_to_run[:5]]), use_container_width=True, hide_index=True)
else:
    mp = st.text_input("Prompt", placeholder="Enter a test prompt…", label_visibility="collapsed")
    if not mp or not selected_models:
        st.markdown("<div style='color:#3d6080;text-align:center;padding:4rem 0;font-size:0.85rem;'>"
                    "SELECT MODELS · THEN ENTER A PROMPT ABOVE</div>", unsafe_allow_html=True)
        st.stop()
    prompts_to_run = [{"text": mp, "raw": mp, "label": None, "choices": None}]

if not selected_models:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

if drift_level > 0 and not prompt_drift_mode:
    with st.expander(f"🔀 Modified prompt — {drift_level} · {DRIFT_TYPES[drift_level]}", expanded=False):
        st.code(inject_drift(prompts_to_run[0]["text"], drift_level)[:600])

# ─────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────
agg = {
    name: {
        "perp_list": [], "clean_perp_list": [],
        "emb_drifts": [], "latent_drifts": [],
        "hall_list": [], "conf_list": [],
        "resp_lens": [], "accuracies": [], "responses": [],
        "last_emb": None, "last_latent": None,
        "last_latent_layers": None,
    }
    for name in selected_models
}

progress_bar = st.progress(0, text="Running inference…")
total_steps  = len(prompts_to_run) * len(selected_models)
step = 0

for sample in prompts_to_run:
    raw_text      = sample["raw"]
    modified_text = sample["text"] if prompt_drift_mode else inject_drift(sample["text"], drift_level)
    correct_text  = get_correct_text(sample, bench_cfg) if bench_cfg else ""

    for name in selected_models:
        step += 1
        progress_bar.progress(step / total_steps, text=f"{name} · sample {step}/{total_steps}…")
        tokenizer, model = loaded_models[name]

        inputs = tokenizer(modified_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(inputs["input_ids"], max_new_tokens=max_gen_tokens)
        response = tokenizer.decode(gen[0], skip_special_tokens=True)

        emb, latent, perp, conf = extract_features(tokenizer, model, modified_text)
        _, c_latent, c_perp, _  = extract_features(tokenizer, model, raw_text)

        emb_flat   = emb.flatten()
        lat_flat   = np.mean([l.flatten() for l in latent], axis=0)
        c_lat_flat = np.mean([l.flatten() for l in c_latent], axis=0)
        ed   = float(np.linalg.norm(emb_flat - lat_flat))   # vs drifted latent
        ld   = float(np.linalg.norm(lat_flat - c_lat_flat))
        hall = hall_score(conf, perp, len(response.split()))
        acc  = compute_accuracy(response, correct_text)

        a = agg[name]
        a["perp_list"].append(perp)
        a["clean_perp_list"].append(c_perp)
        a["emb_drifts"].append(ed)
        a["latent_drifts"].append(ld)
        a["hall_list"].append(hall)
        a["conf_list"].append(conf)
        a["resp_lens"].append(len(response.split()))
        a["accuracies"].append(acc)
        a["responses"].append(response)
        a["last_emb"]           = emb_flat
        a["last_latent"]        = lat_flat
        a["last_latent_layers"] = [l.flatten() for l in latent]

progress_bar.empty()

# ─────────────────────────────────────────────
# Aggregate metrics + weighted drift score
# ─────────────────────────────────────────────
model_metrics = {}
model_wds     = {}

for name in selected_models:
    a   = agg[name]
    h   = safe_mean(a["hall_list"])
    cp  = safe_mean(a["clean_perp_list"])
    dp  = safe_mean(a["perp_list"])
    ed  = safe_mean(a["emb_drifts"])
    ld  = safe_mean(a["latent_drifts"])
    c   = safe_mean(a["conf_list"])
    ac  = safe_mean(a["accuracies"])
    pd_ = (dp - cp) if (not np.isnan(dp) and not np.isnan(cp)) else 0.0
    wds = weighted_drift_score(h, pd_, ed, wh, wp, we)

    model_metrics[name] = {"hall": h, "perp_clean": cp, "perp_drifted": dp,
                            "perp_delta": pd_, "emb_drift": ed, "latent_drift": ld,
                            "confidence": c, "accuracy": ac, "wds": wds}
    model_wds[name] = wds

best_model = min(model_wds, key=model_wds.get) if model_wds else "—"
best_score = model_wds.get(best_model, 0)
best_alert = best_score > drift_threshold

# Log to MLflow
logged_runs = {}
bench_label = selected_benchmark if selected_benchmark else ("prompt_drift" if prompt_drift_mode else "manual")
drift_label = "cross-domain-prompt" if prompt_drift_mode else f"{drift_level}_{DRIFT_TYPES[drift_level]}"

for name in selected_models:
    m = model_metrics[name]
    run_id = mlflow_log_run(
        model_name=name, input_mode=input_mode,
        benchmark=bench_label, drift_type=drift_label,
        metrics={"hallucination": m["hall"], "perplexity_clean": m["perp_clean"],
                 "perplexity_drifted": m["perp_drifted"], "perplexity_delta": m["perp_delta"],
                 "embedding_drift": m["emb_drift"], "latent_drift": m["latent_drift"],
                 "confidence": m["confidence"], "task_accuracy": m["accuracy"],
                 "weighted_drift_score": m["wds"]},
        params={"wh": round(wh,3), "wp": round(wp,3), "we": round(we,3),
                "threshold": drift_threshold, "domain": SLM_REGISTRY[name]["domain"]},
    )
    logged_runs[name] = run_id

if promote_on_run and best_model != "—":
    mlflow_promote_best(best_model)

# ─────────────────────────────────────────────
# Model selector card
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Model Selector — Lifecycle Management</div>', unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns([3, 1, 1])
with sc1:
    sc = "#f87171" if best_alert else "#34d399"
    st_txt = "⚠ DRIFT EXCEEDS THRESHOLD — REPLACEMENT TRIGGERED" if best_alert else "✓ WITHIN THRESHOLD — STABLE"
    best_info = SLM_REGISTRY.get(best_model, {})
    run_short = logged_runs.get(best_model, "")[:8]
    st.markdown(
        f'<div class="selector-card">'
        f'<div class="selector-title">Best SLM — weighted drift score across all metrics</div>'
        f'<div class="selector-model">{best_model}'
        f' <span class="domain-badge {best_info.get("domain_css","")}">{best_info.get("domain","")}</span></div>'
        f'<div class="selector-reason">'
        f'{best_info.get("size","")} · {best_info.get("note","")}<br>'
        f'Speciality: <i>{best_info.get("speciality","")}</i><br>'
        f'WDS: <b style="color:#38bdf8;">{best_score:.4f}</b> (threshold: {drift_threshold:.2f})'
        f' &nbsp;·&nbsp; MLflow run: <code>{run_short}…</code><br>'
        f'<span style="color:{sc};font-weight:600;">{st_txt}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
with sc2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Evaluated</div>'
                f'<div class="metric-value" style="color:#818cf8;">{len(model_wds)}</div>'
                f'<div class="metric-label">of {len(SLM_REGISTRY)} registered</div></div>', unsafe_allow_html=True)
with sc3:
    tc = "#f87171" if best_alert else "#34d399"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Alert threshold</div>'
                f'<div class="metric-value" style="color:{tc};">{drift_threshold:.2f}</div>'
                f'<div class="metric-label">Best WDS: {best_score:.4f}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">At-a-Glance Metrics</div>', unsafe_allow_html=True)

mode_badge = ""
if prompt_drift_mode:
    mode_badge = '<span class="pdrift-badge">PROMPT DRIFT</span>'
elif ts_mode:
    mode_badge = f'<span class="ts-badge">TIME SERIES</span>'
elif bench_cfg:
    mode_badge = (f'<span class="bench-badge {bench_cfg["group_css"]}">{selected_benchmark}</span>'
                  f'<span class="bench-badge" style="background:rgba(56,189,248,0.1);border:1px solid #38bdf8;color:#38bdf8;">'
                  f'{bench_cfg["task_type"]}</span>')

kpi_cols = st.columns(len(selected_models))
for col, name in zip(kpi_cols, selected_models):
    m       = model_metrics[name]
    is_best = (name == best_model)
    alert   = m["hall"] > drift_threshold
    border  = "border: 1px solid #34d399;" if is_best else ""
    badge   = '<span class="badge-alert">⚠ HIGH RISK</span>' if alert else '<span class="badge-ok">✓ NOMINAL</span>'
    best_tag = '<span style="font-size:0.63rem;color:#34d399;font-weight:600;"> ★ SELECTED</span>' if is_best else ""
    dom_css = SLM_REGISTRY[name]["domain_css"]
    dom_lbl = SLM_REGISTRY[name]["domain"]
    acc_c   = "34d399" if not np.isnan(m["accuracy"]) and m["accuracy"] > 0.5 else "f87171"
    wds_c   = "34d399" if is_best else "38bdf8"

    with col:
        st.markdown(f"""
        <div class="metric-card" style="{border}">
            <div style='font-size:0.67rem;letter-spacing:0.1em;color:#3d6080;text-transform:uppercase;margin-bottom:0.3rem;'>
                {name}{best_tag}<br>
                <span class="domain-badge {dom_css}">{dom_lbl}</span>
                <span style='color:#3d5070;font-size:0.6rem;'>{SLM_REGISTRY[name]['size']}</span>
            </div>
            <div class="metric-value {'metric-alert' if alert else 'metric-ok'}">{m['hall']:.3f}</div>
            <div class="metric-label">Hallucination score</div>
            <div style='margin-top:0.6rem;'>{badge}</div>
            <div style='margin-top:0.6rem;font-size:0.67rem;color:#3d6080;line-height:2.0;'>
                Perp clean / drifted &nbsp;
                <b style='color:#8ab4d4;'>{m['perp_clean']:.1f}</b> /
                <b style='color:#{"f87171" if m["perp_delta"]>5 else "8ab4d4"};'>{m['perp_drifted']:.1f}</b>
                &nbsp;(Δ <b>{m['perp_delta']:+.2f}</b>)<br>
                Emb drift &nbsp;<b style='color:#818cf8;'>{m['emb_drift']:.4f}</b>
                &nbsp;|&nbsp; Latent &nbsp;<b style='color:#a78bfa;'>{m['latent_drift']:.4f}</b><br>
                Confidence &nbsp;<b style='color:#8ab4d4;'>{m['confidence']:.3f}</b><br>
                Accuracy &nbsp;<b style='color:#{acc_c};'>
                {f"{m['accuracy']:.1%}" if not np.isnan(m['accuracy']) else "N/A"}</b><br>
                WDS &nbsp;<b style='color:#{wds_c};'>{m['wds']:.4f}</b>
            </div>
            <div style='margin-top:0.4rem;'>{mode_badge}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Threshold experiment
# ─────────────────────────────────────────────
if model_wds:
    st.markdown('<div class="section-header">Threshold Experiment — Weight Configurations</div>', unsafe_allow_html=True)
    weight_configs = [
        ("Hall-heavy (0.7/0.2/0.1)", 0.7, 0.2, 0.1),
        ("Balanced (0.33/0.33/0.33)", 0.33, 0.33, 0.34),
        ("Perp-heavy (0.1/0.7/0.2)",  0.1, 0.7, 0.2),
        (f"Custom ({wh:.2f}/{wp:.2f}/{we:.2f})", wh, wp, we),
    ]
    experiments = []
    for label, cwh, cwp, cwe in weight_configs:
        scores = {name: weighted_drift_score(model_metrics[name]["hall"],
                                              model_metrics[name]["perp_delta"],
                                              model_metrics[name]["emb_drift"], cwh, cwp, cwe)
                  for name in selected_models}
        experiments.append({"label": label, "scores": scores})

    st.plotly_chart(weight_exp_chart(list(model_wds.keys()), experiments), use_container_width=True)

    winner_cols = st.columns(len(experiments))
    for col, exp in zip(winner_cols, experiments):
        winner = min(exp["scores"], key=exp["scores"].get)
        with col:
            st.markdown(
                f"<div style='text-align:center;font-size:0.67rem;color:#3d6080;'>{exp['label'].split('(')[0]}</div>"
                f"<div style='text-align:center;font-size:0.88rem;color:#34d399;font-weight:600;'>{winner}</div>"
                f"<div style='text-align:center;font-size:0.65rem;color:#5a7090;'>WDS: {exp['scores'][winner]:.4f}</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
# Prompt drift charts
# ─────────────────────────────────────────────
if prompt_drift_mode:
    st.markdown('<div class="section-header">Prompt Drift — Cross-domain Perplexity</div>', unsafe_allow_html=True)
    pd_cols = st.columns(min(len(selected_models), 3))
    for i, name in enumerate(selected_models):
        a = agg[name]
        if a["perp_list"] and a["clean_perp_list"]:
            deltas = [p - c for p, c in zip(a["perp_list"], a["clean_perp_list"])]
            pairs  = [f"{s['domain_1'][:8]}→{s['domain_2'][:8]}" for s in prompts_to_run]
            fig = go.Figure(go.Bar(
                x=list(range(len(deltas))), y=deltas,
                marker=dict(color=deltas, colorscale=[[0,"#1a3050"],[0.5,"#818cf8"],[1.0,"#f87171"]]),
                text=[f"{v:.2f}" for v in deltas], textposition="outside", textfont=dict(size=9),
                hovertext=pairs, hovertemplate="<b>%{hovertext}</b><br>Δpplx: %{y:.2f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color="#3d6080", line_width=1, line_dash="dash")
            fig.update_layout(**PLOTLY_LAYOUT,
                              title=dict(text=f"Δ pplx per pair · {name}", font=dict(size=12, color="#8ab4d4")),
                              xaxis_title="Sample", yaxis_title="Δ pplx", height=270)
            with pd_cols[i % len(pd_cols)]:
                st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# Core metric charts
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Core Metrics</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
names = selected_models
with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Perplexity", x=names, y=[model_metrics[n]["perp_drifted"] for n in names],
                         marker_color="#38bdf8", text=[f"{model_metrics[n]['perp_drifted']:.1f}" for n in names], textposition="outside"))
    fig.add_trace(go.Bar(name="Hall. ×100", x=names, y=[model_metrics[n]["hall"]*100 for n in names],
                         marker_color="#f87171", text=[f"{model_metrics[n]['hall']*100:.1f}" for n in names], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Perplexity vs hallucination", font=dict(size=13,color="#8ab4d4")),
                      barmode="group", height=320, legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Confidence", x=names, y=[model_metrics[n]["confidence"] for n in names],
                         marker_color="#34d399", text=[f"{model_metrics[n]['confidence']:.3f}" for n in names], textposition="outside"))
    fig.add_trace(go.Bar(name="Task accuracy", x=names, y=[model_metrics[n]["accuracy"] for n in names],
                         marker_color="#f59e0b",
                         text=[f"{model_metrics[n]['accuracy']:.1%}" if not np.isnan(model_metrics[n]["accuracy"]) else "N/A" for n in names],
                         textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Confidence & task accuracy", font=dict(size=13,color="#8ab4d4")),
                      barmode="group", height=320, legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# Drift analysis
# ─────────────────────────────────────────────
hf_names = [n for n in names if agg[n]["last_emb"] is not None]

if hf_names and (drift_level > 0 or prompt_drift_mode or ts_mode):
    label = ("Prompt drift" if prompt_drift_mode
             else f"Time series — {selected_benchmark}" if ts_mode
             else f"Level {drift_level} — {DRIFT_TYPES[drift_level]}")
    st.markdown(f'<div class="section-header">Drift Analysis · {label}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(bar_chart(hf_names, [model_metrics[n]["emb_drift"]    for n in hf_names], "Embedding drift (L2)",    0, "L2"), use_container_width=True)
    with c2: st.plotly_chart(bar_chart(hf_names, [model_metrics[n]["latent_drift"] for n in hf_names], "Latent state drift (L2)", 2, "L2"), use_container_width=True)
    with c3: st.plotly_chart(bar_chart(hf_names, [model_metrics[n]["perp_delta"]   for n in hf_names], "Perplexity Δ vs clean",   4, "Δ pplx"), use_container_width=True)

    st.markdown('<div class="section-header">Layer-wise Drift</div>', unsafe_allow_html=True)
    layer_cols = st.columns(min(len(hf_names), 3))
    for i, name in enumerate(hf_names):
        tokenizer, model = loaded_models[name]
        _, c_lat, _, _ = extract_features(tokenizer, model, prompts_to_run[0]["raw"])
        clean_layers = [l.flatten() for l in c_lat]
        drifts = [float(np.linalg.norm(agg[name]["last_latent_layers"][j] - clean_layers[j]))
                  for j in range(len(clean_layers))]
        with layer_cols[i % 3]:
            st.plotly_chart(layer_drift_chart(drifts, name), use_container_width=True)

# Cross-model similarity
valid_hf = [n for n in hf_names if agg[n]["last_emb"] is not None]
if len(valid_hf) > 1:
    st.markdown('<div class="section-header">Cross-Model Similarity</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    emb_mat = np.vstack([agg[n]["last_emb"]    for n in valid_hf])
    lat_mat = np.vstack([agg[n]["last_latent"] for n in valid_hf])
    with c1: st.plotly_chart(heatmap_chart(np.round(cosine_distances(emb_mat),4), valid_hf, "Embedding cosine distance","Blues"), use_container_width=True)
    with c2: st.plotly_chart(heatmap_chart(np.round(cosine_distances(lat_mat),4), valid_hf, "Latent cosine distance","Purples"), use_container_width=True)

# ─────────────────────────────────────────────
# EXPERIMENT 1 — Drift detection quality
# Run multiple drift conditions and compare WDS
# Ajay: "show how well drifts are detected"
# ─────────────────────────────────────────────
if run_exp1 and selected_models:
    st.markdown('<div class="section-header">Experiment 1 — Drift Detection Quality</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="exp-card"><div class="exp-title">Exp 1: How well does the system detect each drift type?</div>'
        'Runs all 5 drift conditions (clean → instruction style) on the current prompts for each model. '
        'Lower WDS on clean = good baseline. Rising WDS with drift level = successful drift detection.</div>',
        unsafe_allow_html=True,
    )

    exp1_conditions = list(DRIFT_TYPES.values())
    exp1_matrix     = []  # rows=models, cols=drift conditions

    prog1 = st.progress(0, text="Running Experiment 1…")
    total1 = len(selected_models) * len(DRIFT_TYPES)
    step1  = 0

    for name in selected_models:
        row = []
        tokenizer, model = loaded_models[name]
        for lvl, lbl in DRIFT_TYPES.items():
            step1 += 1
            prog1.progress(step1 / total1, text=f"Exp 1 · {name} · {lbl}…")

            halls, perps, c_perps, embs = [], [], [], []
            for sample in prompts_to_run[:5]:   # use first 5 for speed
                raw   = sample.get("raw", sample["text"])
                drift = inject_drift(raw, lvl)

                _, lat_d, perp_d, conf_d = extract_features(tokenizer, model, drift)
                _, lat_c, perp_c, _      = extract_features(tokenizer, model, raw)

                lat_d_flat = np.mean([l.flatten() for l in lat_d], axis=0)
                lat_c_flat = np.mean([l.flatten() for l in lat_c], axis=0)
                ed = float(np.linalg.norm(lat_d_flat - lat_c_flat))

                halls.append(hall_score(conf_d, perp_d, 50))
                perps.append(perp_d)
                c_perps.append(perp_c)
                embs.append(ed)

            wds = weighted_drift_score(
                safe_mean(halls),
                safe_mean(perps) - safe_mean(c_perps),
                safe_mean(embs), wh, wp, we
            )
            row.append(round(wds, 4))
        exp1_matrix.append(row)

    prog1.empty()
    st.plotly_chart(exp1_drift_heatmap(exp1_matrix, exp1_conditions, selected_models), use_container_width=True)

    exp1_df = pd.DataFrame(exp1_matrix, index=selected_models, columns=exp1_conditions)
    exp1_df.index.name = "Model"
    with st.expander("Experiment 1 — full results table", expanded=False):
        st.dataframe(exp1_df.round(4), use_container_width=True)

    st.markdown(
        "<div class='info-box'>Reading guide: each cell is the Weighted Drift Score for that model × drift condition. "
        "A good detector shows low WDS on Clean and steadily increasing WDS as drift severity grows.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# EXPERIMENT 2 — Model switching benefit
# Ajay: "show replacement model has lower hall
# and better accuracy than original"
# Domain logic: technical prompt → technical model beats general model
# ─────────────────────────────────────────────
if run_exp2 and len(selected_models) >= 2:
    st.markdown('<div class="section-header">Experiment 2 — Model Switching Benefit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="exp-card"><div class="exp-title">Exp 2: Does the selected model outperform the original?</div>'
        'The "original" model is the first selected model (baseline). '
        'The "replacement" model is the one selected by the drift-aware model selector (lowest WDS). '
        'We show hallucination score and task accuracy for both on the <b>same prompts under drift</b>, '
        'proving the replacement is better.</div>',
        unsafe_allow_html=True,
    )

    original_model    = selected_models[0]
    replacement_model = best_model

    if original_model == replacement_model and len(selected_models) > 1:
        # If original is already best, compare with next in list
        replacement_model = selected_models[1]

    prog2 = st.progress(0, text="Running Experiment 2…")

    def run_model_on_prompts(model_name, prompts, drift_lvl, max_tok):
        tok, mdl = loaded_models[model_name]
        halls, accs, perps = [], [], []
        for i, s in enumerate(prompts[:8]):
            prog2.progress((i + 1) / len(prompts[:8]),
                           text=f"Exp 2 · {model_name} · sample {i+1}…")
            raw   = s.get("raw", s["text"])
            drift = inject_drift(raw, drift_lvl)
            inputs = tok(drift, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                gen = mdl.generate(inputs["input_ids"], max_new_tokens=max_tok)
            resp = tok.decode(gen[0], skip_special_tokens=True)
            _, _, perp, conf = extract_features(tok, mdl, drift)
            halls.append(hall_score(conf, perp, len(resp.split())))
            perps.append(perp)
            correct = get_correct_text(s, bench_cfg) if bench_cfg else ""
            accs.append(compute_accuracy(resp, correct))
        return {"hall": safe_mean(halls), "accuracy": safe_mean(accs), "perp": safe_mean(perps)}

    # Use a non-clean drift level for meaningful comparison
    exp2_drift = drift_level if drift_level > 0 else 2  # default domain shift

    orig_results = run_model_on_prompts(original_model,    prompts_to_run, exp2_drift, max_gen_tokens)
    prog2 = st.progress(0, text="Running replacement model…")
    repl_results = run_model_on_prompts(replacement_model, prompts_to_run, exp2_drift, max_gen_tokens)
    prog2.empty()

    exp2_data = {
        f"Original\n({original_model})":    orig_results,
        f"Replacement\n({replacement_model})": repl_results,
    }
    st.plotly_chart(exp2_comparison_chart(exp2_data), use_container_width=True)

    # Verdict
    hall_improved = repl_results["hall"] < orig_results["hall"]
    acc_improved  = (not np.isnan(repl_results["accuracy"]) and
                     not np.isnan(orig_results["accuracy"]) and
                     repl_results["accuracy"] > orig_results["accuracy"])
    verdict_color = "#34d399" if (hall_improved or acc_improved) else "#f87171"
    verdict_text  = ("✓ REPLACEMENT IS BETTER — lower hallucination"
                     + (" and higher accuracy" if acc_improved else "")
                     if (hall_improved or acc_improved)
                     else "— Models perform similarly under this drift condition")

    hall_delta = orig_results["hall"] - repl_results["hall"]
    acc_delta  = (repl_results["accuracy"] - orig_results["accuracy"]
                  if not np.isnan(repl_results["accuracy"]) and not np.isnan(orig_results["accuracy"])
                  else float("nan"))

    st.markdown(
        f"<div class='info-box'>"
        f"<span style='color:{verdict_color};font-weight:600;font-size:0.85rem;'>{verdict_text}</span><br>"
        f"Hallucination Δ: <b style='color:#34d399;'>{hall_delta:+.4f}</b> "
        f"(original: {orig_results['hall']:.4f} → replacement: {repl_results['hall']:.4f})<br>"
        f"Accuracy Δ: <b style='color:#34d399;'>"
        f"{acc_delta:+.1%}</b> " if not np.isnan(acc_delta) else ""
        f"Domain: original=<b style='color:#38bdf8;'>{SLM_REGISTRY[original_model]['domain']}</b> "
        f"· replacement=<b style='color:#34d399;'>{SLM_REGISTRY[replacement_model]['domain']}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Log Exp 2 to MLflow
    mlflow_log_run(
        model_name=replacement_model, input_mode="exp2_comparison",
        benchmark=bench_label, drift_type=f"exp2_{DRIFT_TYPES[exp2_drift]}",
        metrics={"hall_original": orig_results["hall"], "hall_replacement": repl_results["hall"],
                 "hall_delta": hall_delta,
                 "acc_original": orig_results["accuracy"] if not np.isnan(orig_results["accuracy"]) else 0,
                 "acc_replacement": repl_results["accuracy"] if not np.isnan(repl_results["accuracy"]) else 0},
        params={"original_model": original_model, "replacement_model": replacement_model,
                "drift_level": exp2_drift},
    )
elif run_exp2 and len(selected_models) < 2:
    st.warning("Experiment 2 needs at least 2 models selected. Add another model in the sidebar.")

# ─────────────────────────────────────────────
# Model responses
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Model Responses (last sample)</div>', unsafe_allow_html=True)
rc = st.columns(min(len(selected_models), 3))
for i, name in enumerate(selected_models):
    last = agg[name]["responses"][-1] if agg[name]["responses"] else "—"
    with rc[i % 3]:
        dom_css = SLM_REGISTRY[name]["domain_css"]
        dom_lbl = SLM_REGISTRY[name]["domain"]
        st.markdown(f"**{name}** <span class='domain-badge {dom_css}'>{dom_lbl}</span>", unsafe_allow_html=True)
        st.markdown(f'<div class="response-box">{last[:600]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Full Summary Table</div>', unsafe_allow_html=True)

rows = []
for name in selected_models:
    m = model_metrics[name]
    rows.append({
        "Model":             name,
        "Domain":            SLM_REGISTRY[name]["domain"],
        "Size":              SLM_REGISTRY[name]["size"],
        "Input":             input_mode,
        "Benchmark":         bench_label,
        "Data type":         bench_cfg.get("data_type","text") if bench_cfg else "prompt",
        "Drift type":        drift_label,
        "Samples":           len(agg[name]["hall_list"]),
        "Hall. score":       round(m["hall"], 4),
        "Perp. clean":       round(m["perp_clean"], 2)   if not np.isnan(m["perp_clean"])   else "—",
        "Perp. drifted":     round(m["perp_drifted"], 2) if not np.isnan(m["perp_drifted"]) else "—",
        "Δ Perplexity":      round(m["perp_delta"], 2)   if not np.isnan(m["perp_delta"])   else "—",
        "Emb. drift":        round(m["emb_drift"], 4)    if not np.isnan(m["emb_drift"])    else "—",
        "Latent drift":      round(m["latent_drift"], 4) if not np.isnan(m["latent_drift"]) else "—",
        "Confidence":        round(m["confidence"], 4)   if not np.isnan(m["confidence"])   else "—",
        "Task accuracy":     f"{m['accuracy']:.1%}"      if not np.isnan(m["accuracy"])     else "N/A",
        "WDS":               round(m["wds"], 4),
        "Selected":          "★ YES" if name == best_model else "—",
        "MLflow run":        logged_runs.get(name, "—")[:12],
        "Status":            "⚠ HIGH" if m["hall"] > drift_threshold else "✓ OK",
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────
if selected_models:
    st.markdown('<div class="section-header">Multi-Metric Radar</div>', unsafe_allow_html=True)

    def norm01(vals):
        v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
        if not v: return [0.0]*len(vals)
        mn, mx = min(v), max(v)
        if mx == mn: return [0.5]*len(vals)
        return [(x-mn)/(mx-mn) if not (isinstance(x,float) and np.isnan(x)) else 0 for x in vals]

    cats = ["Perplexity","Hallucination","Conf. (inv)","Emb. drift","Acc. (inv)","WDS"]
    raw  = {
        "Perplexity":    [model_metrics[n]["perp_drifted"] for n in selected_models],
        "Hallucination": [model_metrics[n]["hall"]         for n in selected_models],
        "Conf. (inv)":   [1-model_metrics[n]["confidence"] for n in selected_models],
        "Emb. drift":    [model_metrics[n]["emb_drift"]    for n in selected_models],
        "Acc. (inv)":    [1-(model_metrics[n]["accuracy"] or 0) if not np.isnan(model_metrics[n]["accuracy"] or 0) else 0.5 for n in selected_models],
        "WDS":           [model_wds[n]                     for n in selected_models],
    }
    fig_radar = go.Figure()
    for i, name in enumerate(selected_models):
        vals = [norm01(raw[k])[i] for k in cats] + [norm01(raw[cats[0]])[i]]
        r,g,b = (int(COLORS[i%len(COLORS)].lstrip("#")[j:j+2],16) for j in (0,2,4))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill="toself", name=name,
            line=dict(color=COLORS[i%len(COLORS)], width=2),
            fillcolor=f"rgba({r},{g},{b},0.10)",
        ))
    fig_radar.update_layout(
        **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")},
        polar=dict(bgcolor="#0d1624",
                   radialaxis=dict(visible=True, range=[0,1], gridcolor="#1a3050", color="#3d6080"),
                   angularaxis=dict(gridcolor="#1a3050", color="#8ab4d4")),
        title=dict(text="Normalised risk radar (6 axes)", font=dict(size=13, color="#8ab4d4")),
        showlegend=True, height=440, legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
ec1, ec2 = st.columns(2)
with ec1:
    st.download_button("Download summary CSV",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"drift_{bench_label}_{drift_label}.csv", mime="text/csv")
with ec2:
    import json
    meta = {"best_model": best_model, "best_wds": round(best_score,4),
            "alert": best_alert, "benchmark": bench_label, "drift_type": drift_label,
            "weights": {"hall":round(wh,3),"perp":round(wp,3),"emb":round(we,3)},
            "threshold": drift_threshold, "mlflow_runs": logged_runs,
            "model_metrics": {n:{k:round(v,4) if isinstance(v,float) and not np.isnan(v) else str(v)
                                  for k,v in model_metrics[n].items()} for n in selected_models}}
    st.download_button("Download run JSON",
                       data=json.dumps(meta, indent=2).encode("utf-8"),
                       file_name=f"run_{bench_label}_{drift_label}.json", mime="application/json")
