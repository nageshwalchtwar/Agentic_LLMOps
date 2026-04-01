import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_distances
from datasets import load_dataset

# ─────────────────────────────────────────────
# Page config & Global CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Drift Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #080c14;
    color: #c8d8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1420 0%, #0a1018 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #8ab4d4 !important; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 0.75rem;
    color: #3d6080;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #0d1e30 0%, #0a1624 100%);
    border: 1px solid #1a3050;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.65rem;
    color: #3d6080;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.metric-alert { color: #f87171 !important; }
.metric-ok    { color: #34d399 !important; }

/* Model selector card */
.selector-card {
    background: linear-gradient(135deg, #0d2018 0%, #0a1a12 100%);
    border: 1px solid #1a5030;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.selector-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #34d399, #38bdf8);
}
.selector-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    color: #3d8060;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.selector-model {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #34d399;
}
.selector-reason {
    font-size: 0.72rem;
    color: #5a9070;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* Prompt drift badge */
.pdrift-badge {
    display: inline-block;
    background: rgba(248,113,113,0.12);
    border: 1px solid #f87171;
    color: #f87171;
    border-radius: 4px;
    padding: 0.15rem 0.6rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin: 0 2px;
}
.bench-badge {
    display: inline-block;
    border-radius: 4px;
    padding: 0.15rem 0.6rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin: 0 2px;
}
.bench-reasoning     { background: rgba(129,140,248,0.15); border: 1px solid #818cf8; color: #818cf8; }
.bench-understanding { background: rgba(52,211,153,0.15);  border: 1px solid #34d399; color: #34d399; }
.bench-knowledge     { background: rgba(245,158,11,0.15);  border: 1px solid #f59e0b; color: #f59e0b; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-left: 3px solid #38bdf8;
    padding-left: 0.8rem;
    margin: 2rem 0 1rem 0;
}
.response-box {
    background: #0d1624;
    border: 1px solid #1a3050;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    line-height: 1.6;
    color: #a0bcd4;
    max-height: 160px;
    overflow-y: auto;
}
.badge-alert {
    display: inline-block;
    background: rgba(248,113,113,0.15);
    border: 1px solid #f87171;
    color: #f87171;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
}
.badge-ok {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    border: 1px solid #34d399;
    color: #34d399;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
}
.stTextInput > div > div > input {
    background: #0d1624 !important;
    border: 1px solid #1e3a5f !important;
    color: #c8d8f0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
}
hr { border-color: #1a3050 !important; }
[data-testid="stDataFrame"] {
    border: 1px solid #1a3050 !important;
    border-radius: 8px !important;
}
.info-box {
    background: rgba(56,189,248,0.06);
    border: 1px solid #1e3a5f;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: #8ab4d4;
    margin: 0.5rem 0 1rem 0;
    line-height: 1.6;
}
.weight-box {
    background: rgba(56,189,248,0.04);
    border: 1px solid #1a3050;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly dark theme
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#080c14",
    plot_bgcolor="#0d1624",
    font=dict(family="JetBrains Mono", color="#8ab4d4", size=11),
    xaxis=dict(gridcolor="#1a3050", zerolinecolor="#1a3050", showgrid=True),
    yaxis=dict(gridcolor="#1a3050", zerolinecolor="#1a3050", showgrid=True),
    margin=dict(l=50, r=20, t=50, b=40),
)
COLORS = ["#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#f87171", "#a78bfa"]

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# SLM Registry  — 6 models as Ajay requested
# Split into "small" (load on CPU) and "larger"
# so user can pick based on their machine.
# ─────────────────────────────────────────────
SLM_REGISTRY = {
    # ── 2 lightweight models (original) ──────
    "DistilGPT2": {
        "hf_id":  "distilgpt2",
        "type":   "hf",
        "size":   "82M",
        "tier":   "Lightweight",
        "domain": "General",
        "note":   "Fastest. Good clean baseline.",
    },
    "GPT2": {
        "hf_id":  "gpt2",
        "type":   "hf",
        "size":   "117M",
        "tier":   "Lightweight",
        "domain": "General",
        "note":   "Standard GPT-2 small.",
    },
    # ── 4 additional SLMs ─────────────────────
    "GPT2-Medium": {
        "hf_id":  "gpt2-medium",
        "type":   "hf",
        "size":   "345M",
        "tier":   "Medium",
        "domain": "General",
        "note":   "Stronger reasoning baseline.",
    },
    "GPT-Neo-125M": {
        "hf_id":  "EleutherAI/gpt-neo-125M",
        "type":   "hf",
        "size":   "125M",
        "tier":   "Lightweight",
        "domain": "General",
        "note":   "GPT-Neo family. Different arch from GPT2.",
    },
    "OPT-125M": {
        "hf_id":  "facebook/opt-125m",
        "type":   "hf",
        "size":   "125M",
        "tier":   "Lightweight",
        "domain": "General",
        "note":   "Meta OPT. Decoder-only, good for perplexity comparison.",
    },
    "Pythia-160M": {
        "hf_id":  "EleutherAI/pythia-160m",
        "type":   "hf",
        "size":   "160M",
        "tier":   "Lightweight",
        "domain": "General",
        "note":   "Trained on Pile. Good for knowledge drift experiments.",
    },
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
# Prompt Drift — cross-domain question bank
# Ajay: "randomly generate questions on different
#  topics from different domains"
# ─────────────────────────────────────────────
PROMPT_DRIFT_DOMAINS = {
    "Networking / Telecom": [
        "What is the purpose of the TCP/IP protocol stack in modern networks?",
        "Explain how 5G network slicing works and its benefits for operators.",
        "What is the difference between SDN and traditional networking?",
        "How does BGP routing protocol handle path selection?",
        "What are the key components of a TM Forum OSS/BSS architecture?",
        "Explain the role of NFV in modern telecom infrastructure.",
        "What is the difference between LTE and NR in 5G standards?",
        "How does MPLS improve traffic engineering in backbone networks?",
    ],
    "History": [
        "What were the main causes of the First World War?",
        "How did the Renaissance period transform European culture and science?",
        "Describe the economic impact of the Industrial Revolution on society.",
        "What role did the Silk Road play in ancient trade and cultural exchange?",
        "How did the fall of the Roman Empire reshape Medieval Europe?",
        "What were the key factors behind the French Revolution?",
        "Explain the significance of the Magna Carta in legal history.",
        "How did colonialism shape the political boundaries of modern Africa?",
    ],
    "Biology / Medicine": [
        "How does the human immune system respond to a viral infection?",
        "What is the role of mitochondria in cellular energy production?",
        "Explain the mechanism of CRISPR-Cas9 gene editing.",
        "How do vaccines create long-term immunity in the body?",
        "What is the difference between DNA replication and transcription?",
        "Describe the stages of mitosis and their significance.",
        "How does the blood-brain barrier protect the central nervous system?",
        "What are the primary functions of the lymphatic system?",
    ],
    "Economics / Finance": [
        "What is the difference between monetary policy and fiscal policy?",
        "How does inflation affect purchasing power and savings?",
        "Explain the concept of comparative advantage in international trade.",
        "What caused the 2008 global financial crisis?",
        "How do central banks use interest rates to control inflation?",
        "What is the role of the IMF in global economic stability?",
        "Explain the concept of GDP and its limitations as a measure of welfare.",
        "How does quantitative easing affect financial markets?",
    ],
    "Computer Science": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain how a transformer neural network processes sequential data.",
        "What are the trade-offs between time complexity and space complexity?",
        "How does a hash table handle collision resolution?",
        "What is the CAP theorem in distributed systems?",
        "Explain the difference between process and thread in an operating system.",
        "How does public-key cryptography work in secure communication?",
        "What is the role of attention mechanisms in large language models?",
    ],
    "Environmental Science": [
        "How does deforestation contribute to climate change?",
        "What is the role of ocean currents in regulating global temperature?",
        "Explain the greenhouse effect and its impact on Earth's climate.",
        "How do renewable energy sources compare to fossil fuels in carbon output?",
        "What are the main consequences of biodiversity loss in ecosystems?",
        "How does soil erosion affect agricultural productivity?",
        "Explain the water cycle and human activities that disrupt it.",
        "What are the environmental impacts of lithium mining for batteries?",
    ],
    "Philosophy / Ethics": [
        "What is the difference between deontological and consequentialist ethics?",
        "How did Immanuel Kant define the categorical imperative?",
        "What is Plato's allegory of the cave and what does it represent?",
        "Explain the concept of free will versus determinism.",
        "How does utilitarianism approach moral decision making?",
        "What is the trolley problem and why is it important in ethics?",
        "How did Nietzsche's concept of the Übermensch challenge traditional morality?",
        "What is the philosophical difference between rights and duties?",
    ],
    "Physics": [
        "Explain the difference between special and general relativity.",
        "How does quantum entanglement challenge classical physics?",
        "What is the role of dark matter in the structure of the universe?",
        "Explain the Heisenberg uncertainty principle.",
        "How does nuclear fission differ from nuclear fusion?",
        "What is the significance of the Higgs boson in particle physics?",
        "How do black holes form and what happens at the event horizon?",
        "Explain the concept of entropy in thermodynamics.",
    ],
}


def generate_prompt_drift_samples(n: int, seed: int = 42) -> list[dict]:
    """
    Generate n prompt-drift samples by randomly sampling questions
    from two different domains and combining them — simulating
    cross-domain prompt drift as discussed with Ajay.
    """
    rng = random.Random(seed)
    domains = list(PROMPT_DRIFT_DOMAINS.keys())
    samples = []

    for i in range(n):
        # Pick two different domains
        d1, d2 = rng.sample(domains, 2)
        q1 = rng.choice(PROMPT_DRIFT_DOMAINS[d1])
        q2 = rng.choice(PROMPT_DRIFT_DOMAINS[d2])

        # Clean prompt = just domain 1 question
        clean_text = q1

        # Drifted prompt = domain 1 + abrupt domain 2 injection
        drifted_text = f"{q1} Also, {q2.lower()}"

        samples.append({
            "text":        drifted_text,   # this is what gets fed to model
            "raw":         clean_text,     # this is the clean baseline
            "label":       None,
            "choices":     None,
            "domain_1":    d1,
            "domain_2":    d2,
            "question_1":  q1,
            "question_2":  q2,
            "drift_type":  "prompt-drift",
        })

    return samples


# ─────────────────────────────────────────────
# Benchmark registry
# ─────────────────────────────────────────────
BENCHMARK_REGISTRY = {
    "HellaSwag": {
        "hf_path": "Rowan/hellaswag", "hf_name": None, "split": "validation",
        "group": "Reasoning", "group_css": "bench-reasoning",
        "text_col": "ctx", "label_col": "label", "choices_col": "endings",
        "task_type": "sentence-completion-mc",
        "description": "4-way sentence completion. Tests world-knowledge grounded reasoning.",
    },
    "LAMBADA": {
        "hf_path": "EleutherAI/lambada_openai", "hf_name": None, "split": "test",
        "group": "Reasoning", "group_css": "bench-reasoning",
        "text_col": "text", "label_col": None, "choices_col": None,
        "task_type": "sentence-completion",
        "description": "Long-range word prediction. Last word of passage is the answer.",
    },
    "PIQA": {
        "hf_path": "piqa", "hf_name": None, "split": "validation",
        "group": "Reasoning", "group_css": "bench-reasoning",
        "text_col": "goal", "label_col": "label", "choices_col": "sol1,sol2",
        "task_type": "physical-reasoning-mc",
        "description": "Physical intuition QA. 2-choice physical commonsense.",
    },
    "GLUE / SST-2": {
        "hf_path": "glue", "hf_name": "sst2", "split": "validation",
        "group": "Understanding", "group_css": "bench-understanding",
        "text_col": "sentence", "label_col": "label", "choices_col": None,
        "task_type": "sentiment",
        "description": "Binary sentiment classification from movie reviews.",
    },
    "GLUE / MNLI": {
        "hf_path": "glue", "hf_name": "mnli", "split": "validation_matched",
        "group": "Understanding", "group_css": "bench-understanding",
        "text_col": "premise", "label_col": "label", "choices_col": None,
        "task_type": "nli",
        "description": "Natural language inference: premise + hypothesis pairs.",
    },
    "SuperGLUE / BoolQ": {
        "hf_path": "super_glue", "hf_name": "boolq", "split": "validation",
        "group": "Understanding", "group_css": "bench-understanding",
        "text_col": "question", "label_col": "label", "choices_col": None,
        "task_type": "boolean-qa",
        "description": "Yes/no QA with a reading-comprehension passage.",
    },
    "SuperGLUE / CB": {
        "hf_path": "super_glue", "hf_name": "cb", "split": "validation",
        "group": "Understanding", "group_css": "bench-understanding",
        "text_col": "premise", "label_col": "label", "choices_col": None,
        "task_type": "nli",
        "description": "Commitment bank: 3-class textual entailment.",
    },
    "MMLU / abstract_algebra": {
        "hf_path": "cais/mmlu", "hf_name": "abstract_algebra", "split": "test",
        "group": "Knowledge", "group_css": "bench-knowledge",
        "text_col": "question", "label_col": "answer", "choices_col": "choices",
        "task_type": "knowledge-mc",
        "description": "MMLU abstract algebra. 4-way MCQ.",
    },
    "MMLU / anatomy": {
        "hf_path": "cais/mmlu", "hf_name": "anatomy", "split": "test",
        "group": "Knowledge", "group_css": "bench-knowledge",
        "text_col": "question", "label_col": "answer", "choices_col": "choices",
        "task_type": "knowledge-mc",
        "description": "MMLU anatomy. Tests biomedical knowledge.",
    },
    "MMLU / high_school_history": {
        "hf_path": "cais/mmlu", "hf_name": "high_school_us_history", "split": "test",
        "group": "Knowledge", "group_css": "bench-knowledge",
        "text_col": "question", "label_col": "answer", "choices_col": "choices",
        "task_type": "knowledge-mc",
        "description": "MMLU US history. Tests world-knowledge under domain shift.",
    },
    "MMLU / computer_security": {
        "hf_path": "cais/mmlu", "hf_name": "computer_security", "split": "test",
        "group": "Knowledge", "group_css": "bench-knowledge",
        "text_col": "question", "label_col": "answer", "choices_col": "choices",
        "task_type": "knowledge-mc",
        "description": "MMLU computer security. Closest to telecom/tech domain.",
    },
}

GROK_API_KEY = "xai-denf0c3j0T0DuWnJw6TdhghG5vuJVPP3UTcu3uJBQYDeAkLgulUKP2tuLcDZcn51EGYBJ9WWUyT4fzF1"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Configuration")

    # ── SLM selection ─────────────────────────
    st.markdown("**SLM Registry** — select models to evaluate")

    tier_filter = st.radio(
        "Filter by tier",
        ["Lightweight only (≤125M)", "All models (up to 345M)"],
        index=0,
        horizontal=False,
    )
    if "Lightweight" in tier_filter:
        available_models = [k for k, v in SLM_REGISTRY.items() if v["tier"] == "Lightweight"]
    else:
        available_models = list(SLM_REGISTRY.keys())

    selected_models = st.multiselect(
        "Models",
        available_models,
        default=available_models[:2],
        format_func=lambda x: f"{x}  [{SLM_REGISTRY[x]['size']}]",
    )

    # Show registry info
    if selected_models:
        info_lines = []
        for m in selected_models:
            r = SLM_REGISTRY[m]
            info_lines.append(
                f"<span style='color:#38bdf8'>{m}</span> "
                f"<span style='color:#3d6080'>{r['size']} · {r['note']}</span>"
            )
        st.markdown(
            "<div style='font-size:0.68rem;line-height:1.8;margin-top:0.3rem;'>"
            + "<br>".join(info_lines)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    drift_type_options = list(DRIFT_TYPES.items())
    drift_level = st.select_slider(
        "Drift injection type",
        options=[d[0] for d in drift_type_options],
        format_func=lambda x: f"{x} · {DRIFT_TYPES[x]}",
        value=0,
    )
    max_gen_tokens = st.slider("Max generation tokens", 50, 300, 150, step=25)

    st.markdown("**Device:** `" + DEVICE + "`")

    # ── Dataset / Input Panel ─────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Input Panel")

    input_mode = st.radio(
        "Input mode",
        ["Benchmark dataset", "Prompt drift (cross-domain)", "Manual prompt"],
        index=0,
    )

    selected_benchmark = None
    bench_cfg          = None
    num_samples        = 10
    prompt_drift_seed  = 42

    if input_mode == "Benchmark dataset":
        bench_names = list(BENCHMARK_REGISTRY.keys())
        selected_benchmark = st.selectbox(
            "Benchmark",
            bench_names,
            format_func=lambda x: f"{x}  [{BENCHMARK_REGISTRY[x]['group']}]",
        )
        bench_cfg   = BENCHMARK_REGISTRY[selected_benchmark]
        num_samples = st.slider("Samples per run", 5, 50, 15, step=5)
        group_color = {"Reasoning": "#818cf8", "Understanding": "#34d399", "Knowledge": "#f59e0b"}.get(
            bench_cfg["group"], "#8ab4d4"
        )
        st.markdown(
            f"<div style='font-size:0.72rem;margin-top:0.4rem;'>"
            f"<span style='color:{group_color};font-weight:600;'>{bench_cfg['group']}</span>"
            f" · <span style='color:#5a7a94;'>{bench_cfg['task_type']}</span><br>"
            f"<span style='color:#3d6080;'>{bench_cfg['description']}</span></div>",
            unsafe_allow_html=True,
        )

    elif input_mode == "Prompt drift (cross-domain)":
        num_samples       = st.slider("Number of prompt pairs", 5, 50, 10, step=5)
        prompt_drift_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        domain_list = list(PROMPT_DRIFT_DOMAINS.keys())
        st.markdown(
            "<div style='font-size:0.7rem;color:#3d6080;margin-top:0.4rem;line-height:1.7;'>"
            f"Randomly samples questions from 2 different domains per pair.<br>"
            f"<b style='color:#8ab4d4;'>Domains:</b> {', '.join(domain_list)}</div>",
            unsafe_allow_html=True,
        )

    # ── Weighted threshold experiment ─────────
    # Ajay: "show different experiments with weighted mean of metrics"
    st.markdown("---")
    st.markdown("### ⚖️ Model Selector Weights")
    st.markdown(
        "<div style='font-size:0.68rem;color:#3d6080;margin-bottom:0.5rem;'>"
        "Weighted drift score used to select best SLM.<br>"
        "Weights must sum to 1.0 (auto-normalised).</div>",
        unsafe_allow_html=True,
    )
    w_hall = st.slider("Hallucination weight",   0.0, 1.0, 0.5, step=0.05)
    w_perp = st.slider("Perplexity Δ weight",    0.0, 1.0, 0.3, step=0.05)
    w_emb  = st.slider("Embedding drift weight", 0.0, 1.0, 0.2, step=0.05)
    total_w = w_hall + w_perp + w_emb
    if total_w == 0:
        total_w = 1.0
    w_hall_n = w_hall / total_w
    w_perp_n = w_perp / total_w
    w_emb_n  = w_emb  / total_w

    drift_threshold = st.slider("Alert threshold", 0.1, 1.0, 0.5, step=0.05)
    st.markdown(
        f"<div style='font-size:0.68rem;color:#3d6080;'>Normalised — Hall: "
        f"<b style='color:#f87171;'>{w_hall_n:.2f}</b> · "
        f"Perp: <b style='color:#38bdf8;'>{w_perp_n:.2f}</b> · "
        f"Emb: <b style='color:#818cf8;'>{w_emb_n:.2f}</b></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Neural Drift Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Agentic lifecycle management · drift detection · SLM benchmarking</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
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


loaded_models = {}
for name in selected_models:
    info = SLM_REGISTRY[name]
    if info["type"] == "hf":
        with st.spinner(f"Loading {name} ({info['size']})…"):
            loaded_models[name] = load_model(info["hf_id"])

# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_benchmark_samples(benchmark_name: str, n: int) -> list[dict]:
    cfg = BENCHMARK_REGISTRY[benchmark_name]
    load_kwargs = {"path": cfg["hf_path"], "split": cfg["split"]}
    if cfg.get("hf_name"):
        load_kwargs["name"] = cfg["hf_name"]
    try:
        ds = load_dataset(**load_kwargs)
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    except Exception as e:
        st.error(f"Failed to load {benchmark_name}: {e}")
        return []
    samples = []
    for row in ds:
        raw_text = str(row.get(cfg["text_col"], ""))
        label    = row.get(cfg["label_col"]) if cfg["label_col"] else None
        choices  = None
        if cfg["choices_col"]:
            if "," in cfg["choices_col"]:
                cols    = [c.strip() for c in cfg["choices_col"].split(",")]
                choices = [str(row.get(c, "")) for c in cols]
            else:
                choices = list(row.get(cfg["choices_col"], []))
        if choices:
            options_str = "\n".join(f"({i}) {c}" for i, c in enumerate(choices))
            prompt_text = f"{raw_text}\n{options_str}"
        else:
            prompt_text = raw_text
        samples.append({"text": prompt_text, "raw": raw_text, "label": label, "choices": choices})
    return samples


def get_correct_text(sample: dict, cfg: dict) -> str:
    if sample.get("label") is None or not sample.get("choices"):
        return ""
    try:
        return sample["choices"][int(sample["label"])]
    except Exception:
        return str(sample["label"])


def compute_accuracy(generated: str, correct: str) -> float:
    if not correct:
        return float("nan")
    gen_t = set(generated.lower().split())
    cor_t = set(correct.lower().split())
    return len(gen_t & cor_t) / len(cor_t) if cor_t else 0.0


# ─────────────────────────────────────────────
# Drift injection
# ─────────────────────────────────────────────
def inject_drift(prompt: str, level: int) -> str:
    if level == 0:  return prompt
    if level == 1:  return prompt + " Also consider unrelated context about streaming services."
    if level == 2:  return prompt + " Also discuss Roman law history."
    if level == 3:  return prompt + " Also summarize climate change policies."
    if level == 4:  return (prompt + " ") * 30 + "Please provide a detailed answer."
    if level == 5:  return "Write a 500 word detailed research explanation about: " + prompt + " Include examples and references."
    return prompt


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────
def extract_features(tokenizer, model, prompt: str):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden  = outputs.hidden_states[-1]
    embedding    = last_hidden.mean(dim=1).cpu().numpy()
    latent       = [h.mean(dim=1).cpu().numpy() for h in outputs.hidden_states]
    logits       = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    perplexity = torch.exp(loss).item()
    probs      = torch.softmax(logits, dim=-1)
    confidence = torch.max(probs, dim=-1)[0].mean().item()
    return embedding, latent, perplexity, confidence, logits


def hall_score(confidence, perplexity, response_len):
    if not np.isnan(confidence):
        return (1 - confidence) * 0.5 + (perplexity / 100) * 0.3 + (response_len / 200) * 0.2
    return (response_len / 200) * 0.2


def safe_mean(lst):
    vals = [v for v in lst if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


# ─────────────────────────────────────────────
# Weighted drift score — Ajay's model selector
# ─────────────────────────────────────────────
def weighted_drift_score(hall, perp_delta, emb_drift, wh, wp, we):
    """
    Composite drift score used to rank SLMs.
    Lower = better (less drift / hallucination).
    perp_delta is normalised to [0,1] range by dividing by 50.
    emb_drift  is normalised by dividing by 5.
    """
    norm_perp = min(abs(perp_delta) / 50.0, 1.0)
    norm_emb  = min(emb_drift / 5.0, 1.0)
    norm_hall = min(hall, 1.0)
    return wh * norm_hall + wp * norm_perp + we * norm_emb


# ─────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────
def bar_chart(names, values, title, color_idx=0, unit=""):
    safe_vals = [v if (v is not None and not np.isnan(v)) else 0 for v in values]
    fig = go.Figure(go.Bar(
        x=names, y=safe_vals,
        marker=dict(
            color=safe_vals,
            colorscale=[[0, "#1a3050"], [0.5, COLORS[color_idx]], [1.0, COLORS[(color_idx + 1) % len(COLORS)]]],
            line=dict(color="#080c14", width=1),
        ),
        text=[f"{v:.3f}" for v in safe_vals],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#8ab4d4")),
        yaxis_title=unit, showlegend=False, height=300,
    )
    return fig


def heatmap_chart(matrix, labels, title, colorscale="Blues"):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale=colorscale,
        text=np.round(matrix, 4), texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#8ab4d4")), height=320,
    )
    return fig


def layer_drift_chart(drift_per_layer, name):
    layers = list(range(len(drift_per_layer)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=drift_per_layer, mode="lines+markers",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=6, color="#818cf8"),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
    ))
    fig.add_trace(go.Bar(
        x=layers, y=drift_per_layer,
        marker=dict(color=drift_per_layer, colorscale=[[0, "#1a3050"], [1, "#f87171"]], opacity=0.35),
        showlegend=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Layer-wise drift · {name}", font=dict(size=13, color="#8ab4d4")),
        xaxis_title="Layer", yaxis_title="L2 drift", height=250,
    )
    return fig


def weighted_score_experiment_chart(model_scores: dict, experiments: list[dict]):
    """
    Show how model ranking changes under different weight configurations.
    experiments: list of {label, scores_dict} where scores_dict maps model->score
    """
    fig = go.Figure()
    models = list(model_scores.keys())
    for i, exp in enumerate(experiments):
        fig.add_trace(go.Bar(
            name=exp["label"],
            x=models,
            y=[exp["scores"].get(m, 0) for m in models],
            marker_color=COLORS[i % len(COLORS)],
            text=[f"{exp['scores'].get(m, 0):.3f}" for m in models],
            textposition="outside",
            textfont=dict(size=9),
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Weighted drift score — threshold experiments", font=dict(size=13, color="#8ab4d4")),
        barmode="group", height=340,
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def prompt_drift_domain_chart(samples: list[dict], model_name: str, perp_deltas: list[float]):
    """Chart showing perplexity delta per domain pair — for prompt drift mode."""
    domain_pairs = [f"{s['domain_1'][:8]}→{s['domain_2'][:8]}" for s in samples]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(perp_deltas))),
        y=perp_deltas,
        marker=dict(
            color=perp_deltas,
            colorscale=[[0, "#1a3050"], [0.5, "#818cf8"], [1.0, "#f87171"]],
        ),
        text=[f"{v:.2f}" for v in perp_deltas],
        textposition="outside",
        textfont=dict(size=9),
        hovertext=domain_pairs,
        hovertemplate="<b>%{hovertext}</b><br>Δ pplx: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#3d6080", line_width=1, line_dash="dash")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Δ perplexity per cross-domain pair · {model_name}", font=dict(size=13, color="#8ab4d4")),
        xaxis_title="Sample (hover for domains)",
        yaxis_title="Δ pplx",
        height=280,
    )
    return fig


# ─────────────────────────────────────────────
# Build prompt list
# ─────────────────────────────────────────────
prompts_to_run = []
prompt_drift_mode = False

if input_mode == "Benchmark dataset":
    st.markdown(
        f'<div class="info-box">'
        f'<b style="color:#38bdf8">{selected_benchmark}</b>'
        f' [{bench_cfg["group"]}] — {bench_cfg["description"]}<br>'
        f'Loading <b>{num_samples}</b> samples · task: <code>{bench_cfg["task_type"]}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.spinner(f"Loading {selected_benchmark}…"):
        prompts_to_run = load_benchmark_samples(selected_benchmark, num_samples)
    if not prompts_to_run:
        st.error("No samples loaded.")
        st.stop()
    manual_override = st.text_input(
        "Override with single custom prompt (leave blank for benchmark)",
        placeholder="Optional…",
    )
    if manual_override.strip():
        prompts_to_run = [{"text": manual_override, "raw": manual_override, "label": None, "choices": None}]

elif input_mode == "Prompt drift (cross-domain)":
    prompt_drift_mode = True
    st.markdown(
        f'<div class="info-box">'
        f'<span class="pdrift-badge">PROMPT DRIFT</span>'
        f' Generating <b>{num_samples}</b> cross-domain question pairs · seed <b>{prompt_drift_seed}</b><br>'
        f'Each sample mixes two unrelated domains to simulate real prompt distribution shift.'
        f'</div>',
        unsafe_allow_html=True,
    )
    prompts_to_run = generate_prompt_drift_samples(num_samples, seed=int(prompt_drift_seed))

    # Show sample of generated pairs
    with st.expander("Preview generated prompt-drift pairs", expanded=False):
        preview_df = pd.DataFrame([
            {
                "Domain 1":   s["domain_1"],
                "Domain 2":   s["domain_2"],
                "Clean prompt (Q1)":   s["question_1"],
                "Drift injection (Q2)": s["question_2"],
            }
            for s in prompts_to_run[:5]
        ])
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

else:
    manual_prompt = st.text_input("Prompt", placeholder="Enter a test prompt…", label_visibility="collapsed")
    if not manual_prompt or not selected_models:
        st.markdown(
            "<div style='color:#3d6080;text-align:center;padding:4rem 0;"
            "font-size:0.85rem;letter-spacing:0.1em;'>"
            "SELECT MODELS · THEN ENTER A PROMPT ABOVE</div>",
            unsafe_allow_html=True,
        )
        st.stop()
    prompts_to_run = [{"text": manual_prompt, "raw": manual_prompt, "label": None, "choices": None}]

if not selected_models:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

# Drift preview (non-prompt-drift modes)
if drift_level > 0 and not prompt_drift_mode:
    example_modified = inject_drift(prompts_to_run[0]["text"], drift_level)
    with st.expander(f"🔀 Modified prompt — level {drift_level} ({DRIFT_TYPES[drift_level]})", expanded=False):
        st.code(example_modified[:800] + ("…" if len(example_modified) > 800 else ""))

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
        "last_clean_emb": None, "last_clean_latent": None,
        "last_latent_layers": None,
    }
    for name in selected_models
}

progress_bar = st.progress(0, text="Running inference…")
total_steps  = len(prompts_to_run) * len(selected_models)
step         = 0

for sample in prompts_to_run:
    # For prompt drift mode, raw=clean question, text=drifted pair
    # For benchmark/manual, apply inject_drift on top
    if prompt_drift_mode:
        raw_text      = sample["raw"]
        modified_text = sample["text"]
    else:
        raw_text      = sample["text"]
        modified_text = inject_drift(raw_text, drift_level)

    correct_text = get_correct_text(sample, bench_cfg) if bench_cfg else ""

    for name in selected_models:
        step += 1
        progress_bar.progress(step / total_steps, text=f"{name} · sample {step}/{total_steps}…")

        info = SLM_REGISTRY[name]
        if info["type"] == "hf":
            tokenizer, model = loaded_models[name]

            inputs = tokenizer(
                modified_text, return_tensors="pt", truncation=True, max_length=512
            ).to(DEVICE)
            with torch.no_grad():
                gen = model.generate(inputs["input_ids"], max_new_tokens=max_gen_tokens)
            response = tokenizer.decode(gen[0], skip_special_tokens=True)

            emb, latent, perp, conf, _ = extract_features(tokenizer, model, modified_text)
            c_emb, c_latent, c_perp, _, _ = extract_features(tokenizer, model, raw_text)

            emb_flat   = emb.flatten()
            c_emb_flat = c_emb.flatten()
            lat_flat   = np.mean([l.flatten() for l in latent], axis=0)
            c_lat_flat = np.mean([l.flatten() for l in c_latent], axis=0)

            emb_drift    = float(np.linalg.norm(emb_flat - c_emb_flat))
            latent_drift = float(np.linalg.norm(lat_flat - c_lat_flat))
            hall         = hall_score(conf, perp, len(response.split()))
            acc          = compute_accuracy(response, correct_text)

            a = agg[name]
            a["perp_list"].append(perp)
            a["clean_perp_list"].append(c_perp)
            a["emb_drifts"].append(emb_drift)
            a["latent_drifts"].append(latent_drift)
            a["hall_list"].append(hall)
            a["conf_list"].append(conf)
            a["resp_lens"].append(len(response.split()))
            a["accuracies"].append(acc)
            a["responses"].append(response)
            a["last_emb"]           = emb_flat
            a["last_latent"]        = lat_flat
            a["last_clean_emb"]     = c_emb_flat
            a["last_clean_latent"]  = c_lat_flat
            a["last_latent_layers"] = [l.flatten() for l in latent]

progress_bar.empty()

# ─────────────────────────────────────────────
# Model selector — Ajay's lifecycle component
# Picks best SLM based on weighted drift score
# ─────────────────────────────────────────────
model_scores    = {}
model_wds       = {}

for name in selected_models:
    a = agg[name]
    if not a["hall_list"]:
        continue
    h  = safe_mean(a["hall_list"])
    pd_ = safe_mean(a["perp_list"]) - safe_mean(a["clean_perp_list"])
    ed = safe_mean(a["emb_drifts"])
    wds = weighted_drift_score(h, pd_, ed, w_hall_n, w_perp_n, w_emb_n)
    model_scores[name] = wds
    model_wds[name]    = {"hall": h, "perp_delta": pd_, "emb_drift": ed}

best_model    = min(model_scores, key=model_scores.get) if model_scores else "—"
best_score    = model_scores.get(best_model, 0)
best_alert    = best_score > drift_threshold
best_slm_info = SLM_REGISTRY.get(best_model, {})

# ─────────────────────────────────────────────
# Model selector card
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Model Selector — Lifecycle Management</div>', unsafe_allow_html=True)

sel_col1, sel_col2, sel_col3 = st.columns([2, 1, 1])

with sel_col1:
    status_color = "#f87171" if best_alert else "#34d399"
    status_text  = "⚠ DRIFT EXCEEDS THRESHOLD — RECOMMEND REPLACEMENT" if best_alert else "✓ WITHIN THRESHOLD — MODEL STABLE"
    st.markdown(
        f'<div class="selector-card">'
        f'<div class="selector-title">Best SLM selected by weighted drift score</div>'
        f'<div class="selector-model">{best_model}</div>'
        f'<div class="selector-reason">'
        f'{best_slm_info.get("size", "")} · {best_slm_info.get("note", "")}<br>'
        f'Weighted drift score: <b style="color:#38bdf8;">{best_score:.4f}</b> '
        f'(threshold: {drift_threshold:.2f})<br>'
        f'<span style="color:{status_color};font-weight:600;">{status_text}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

with sel_col2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Models evaluated</div>'
        f'<div class="metric-value" style="color:#818cf8;">{len(model_scores)}</div>'
        f'<div class="metric-label">SLMs in registry: {len(SLM_REGISTRY)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with sel_col3:
    threshold_status_color = "#f87171" if best_alert else "#34d399"
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Drift threshold</div>'
        f'<div class="metric-value" style="color:{threshold_status_color};">{drift_threshold:.2f}</div>'
        f'<div class="metric-label">Best score: {best_score:.4f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">At-a-Glance Metrics</div>', unsafe_allow_html=True)

mode_badge = ""
if prompt_drift_mode:
    mode_badge = '<span class="pdrift-badge">PROMPT DRIFT</span>'
elif bench_cfg:
    css  = bench_cfg["group_css"]
    mode_badge = (
        f'<span class="bench-badge {css}">{selected_benchmark}</span>'
        f'<span class="bench-badge" style="background:rgba(56,189,248,0.1);border:1px solid #38bdf8;color:#38bdf8;">'
        f'{bench_cfg["task_type"]}</span>'
    )

kpi_cols = st.columns(len(selected_models))
for col, name in zip(kpi_cols, selected_models):
    a     = agg[name]
    h     = safe_mean(a["hall_list"])
    p     = safe_mean(a["perp_list"])
    c     = safe_mean(a["conf_list"])
    ed    = safe_mean(a["emb_drifts"])
    acc   = safe_mean(a["accuracies"])
    wds   = model_scores.get(name, float("nan"))
    is_best = (name == best_model)
    alert   = h > drift_threshold
    border_extra = "border: 1px solid #34d399;" if is_best else ""
    badge = '<span class="badge-alert">⚠ HIGH RISK</span>' if alert else '<span class="badge-ok">✓ NOMINAL</span>'
    best_tag = '<span style="font-size:0.65rem;color:#34d399;font-weight:600;"> ★ SELECTED</span>' if is_best else ""

    with col:
        st.markdown(f"""
        <div class="metric-card" style="{border_extra}">
            <div style='font-size:0.7rem;letter-spacing:0.12em;color:#3d6080;text-transform:uppercase;margin-bottom:0.3rem;'>
                {name} {best_tag}
                <span style='font-size:0.62rem;color:#3d5070;'>({SLM_REGISTRY[name]['size']})</span>
            </div>
            <div class="metric-value {'metric-alert' if alert else 'metric-ok'}">{h:.3f}</div>
            <div class="metric-label">Hallucination score (mean)</div>
            <div style='margin-top:0.7rem;'>{badge}</div>
            <div style='margin-top:0.7rem;font-size:0.68rem;color:#3d6080;line-height:1.9;'>
                Perplexity &nbsp;<b style='color:#8ab4d4;'>{p:.1f if not np.isnan(p) else "N/A"}</b> &nbsp;|&nbsp;
                Conf &nbsp;<b style='color:#8ab4d4;'>{c:.3f if not np.isnan(c) else "N/A"}</b><br>
                Emb drift &nbsp;<b style='color:#8ab4d4;'>{ed:.4f if not np.isnan(ed) else "N/A"}</b> &nbsp;|&nbsp;
                n=<b style='color:#8ab4d4;'>{len(a["hall_list"])}</b><br>
                Task accuracy &nbsp;<b style='color:#{"34d399" if not np.isnan(acc) and acc > 0.5 else "f87171"};'>
                {f"{acc:.1%}" if not np.isnan(acc) else "N/A"}</b><br>
                <span style='color:#5a7090;'>Weighted score: </span>
                <b style='color:#{"34d399" if is_best else "38bdf8"};'>{wds:.4f if not np.isnan(wds) else "N/A"}</b>
            </div>
            <div style='margin-top:0.5rem;'>{mode_badge}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Threshold experiment chart
# Ajay: "show different experiments with weighted means"
# ─────────────────────────────────────────────
if len(model_scores) > 0:
    st.markdown('<div class="section-header">Threshold Experiment — Weight Configurations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Comparing how model ranking changes under 3 weighting strategies. '
        'Lower score = less drift = preferred model for selection.</div>',
        unsafe_allow_html=True,
    )

    experiments = []
    weight_configs = [
        ("Hallucination-heavy (0.7/0.2/0.1)", 0.7, 0.2, 0.1),
        ("Balanced (0.33/0.33/0.33)",          0.33, 0.33, 0.33),
        ("Perplexity-heavy (0.1/0.7/0.2)",     0.1, 0.7, 0.2),
        (f"Custom ({w_hall_n:.2f}/{w_perp_n:.2f}/{w_emb_n:.2f})", w_hall_n, w_perp_n, w_emb_n),
    ]
    for label, wh, wp, we in weight_configs:
        scores = {}
        for name in selected_models:
            if name not in model_wds:
                continue
            d = model_wds[name]
            scores[name] = weighted_drift_score(d["hall"], d["perp_delta"], d["emb_drift"], wh, wp, we)
        experiments.append({"label": label, "scores": scores})

    st.plotly_chart(weighted_score_experiment_chart(model_scores, experiments), use_container_width=True)

    # Show winner per config
    winner_cols = st.columns(len(experiments))
    for col, exp in zip(winner_cols, experiments):
        if exp["scores"]:
            winner = min(exp["scores"], key=exp["scores"].get)
            score  = exp["scores"][winner]
            with col:
                st.markdown(
                    f"<div style='text-align:center;font-size:0.7rem;color:#3d6080;margin-bottom:0.2rem;'>"
                    f"{exp['label'].split('(')[0].strip()}</div>"
                    f"<div style='text-align:center;font-size:0.9rem;color:#34d399;font-weight:600;'>{winner}</div>"
                    f"<div style='text-align:center;font-size:0.68rem;color:#5a7090;'>score: {score:.4f}</div>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────
# Prompt drift specific charts
# ─────────────────────────────────────────────
if prompt_drift_mode:
    st.markdown('<div class="section-header">Prompt Drift — Cross-domain Analysis</div>', unsafe_allow_html=True)

    pd_cols = st.columns(min(len(selected_models), 3))
    for i, name in enumerate(selected_models):
        a = agg[name]
        if a["perp_list"] and a["clean_perp_list"]:
            deltas = [p - c for p, c in zip(a["perp_list"], a["clean_perp_list"])]
            with pd_cols[i % len(pd_cols)]:
                st.plotly_chart(
                    prompt_drift_domain_chart(prompts_to_run, name, deltas),
                    use_container_width=True,
                )

    # Domain pair table
    with st.expander("Full prompt-drift sample table", expanded=False):
        rows_pd = []
        for j, s in enumerate(prompts_to_run):
            row = {
                "Sample": j,
                "Domain 1": s["domain_1"],
                "Domain 2": s["domain_2"],
                "Clean prompt": s["question_1"][:60] + "…",
                "Drift injection": s["question_2"][:60] + "…",
            }
            for name in selected_models:
                a = agg[name]
                if j < len(a["perp_list"]) and j < len(a["clean_perp_list"]):
                    row[f"{name} Δpplx"] = round(a["perp_list"][j] - a["clean_perp_list"][j], 2)
            rows_pd.append(row)
        st.dataframe(pd.DataFrame(rows_pd), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Model responses
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Model Responses (last sample)</div>', unsafe_allow_html=True)
resp_cols = st.columns(min(len(selected_models), 3))
for i, name in enumerate(selected_models):
    a = agg[name]
    last_resp = a["responses"][-1] if a["responses"] else "—"
    with resp_cols[i % 3]:
        st.markdown(f"**{name}**")
        st.markdown(f'<div class="response-box">{last_resp[:600]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Core metrics charts
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Core Metrics</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

names = selected_models
perps = [safe_mean(agg[n]["perp_list"]) for n in names]
halls = [safe_mean(agg[n]["hall_list"]) for n in names]
confs = [safe_mean(agg[n]["conf_list"]) for n in names]
accs  = [safe_mean(agg[n]["accuracies"]) for n in names]

with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Perplexity", x=names, y=perps, marker_color="#38bdf8",
                         text=[f"{v:.1f}" for v in perps], textposition="outside"))
    fig.add_trace(go.Bar(name="Hall. ×100",  x=names, y=[h * 100 for h in halls],
                         marker_color="#f87171",
                         text=[f"{v:.1f}" for v in [h * 100 for h in halls]], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Perplexity vs hallucination score", font=dict(size=13, color="#8ab4d4")),
                      barmode="group", height=320, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Confidence", x=names, y=confs, marker_color="#34d399",
                         text=[f"{v:.3f}" for v in confs], textposition="outside"))
    fig.add_trace(go.Bar(name="Task accuracy", x=names, y=accs, marker_color="#f59e0b",
                         text=[f"{v:.1%}" if not np.isnan(v) else "N/A" for v in accs],
                         textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Confidence & task accuracy", font=dict(size=13, color="#8ab4d4")),
                      barmode="group", height=320, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# Drift analysis
# ─────────────────────────────────────────────
hf_names = [n for n in names if agg[n]["last_emb"] is not None]

if hf_names and (drift_level > 0 or prompt_drift_mode):
    label = "Prompt drift" if prompt_drift_mode else f"Level {drift_level} — {DRIFT_TYPES[drift_level]}"
    st.markdown(f'<div class="section-header">Drift Analysis · {label}</div>', unsafe_allow_html=True)

    emb_drifts    = [safe_mean(agg[n]["emb_drifts"])    for n in hf_names]
    latent_drifts = [safe_mean(agg[n]["latent_drifts"]) for n in hf_names]
    perp_deltas   = [safe_mean(agg[n]["perp_list"]) - safe_mean(agg[n]["clean_perp_list"]) for n in hf_names]

    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(bar_chart(hf_names, emb_drifts,    "Embedding drift (L2)",    0, "L2 norm"), use_container_width=True)
    with c2: st.plotly_chart(bar_chart(hf_names, latent_drifts, "Latent state drift (L2)", 2, "L2 norm"), use_container_width=True)
    with c3: st.plotly_chart(bar_chart(hf_names, perp_deltas,   "Perplexity Δ vs clean",   4, "Δ pplx"),  use_container_width=True)

    # Layer-wise
    st.markdown('<div class="section-header">Layer-wise Drift</div>', unsafe_allow_html=True)
    layer_cols = st.columns(min(len(hf_names), 3))
    for i, name in enumerate(hf_names):
        tokenizer, model = loaded_models[name]
        _, c_lat, _, _, _ = extract_features(tokenizer, model, prompts_to_run[0]["raw"])
        clean_layers = [l.flatten() for l in c_lat]
        drifts = [float(np.linalg.norm(agg[name]["last_latent_layers"][j] - clean_layers[j]))
                  for j in range(len(clean_layers))]
        with layer_cols[i % 3]:
            st.plotly_chart(layer_drift_chart(drifts, name), use_container_width=True)

# ─────────────────────────────────────────────
# Cross-model similarity heatmaps
# ─────────────────────────────────────────────
valid_hf = [n for n in hf_names if agg[n]["last_emb"] is not None]
if len(valid_hf) > 1:
    st.markdown('<div class="section-header">Cross-Model Similarity</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    emb_matrix    = np.vstack([agg[n]["last_emb"]    for n in valid_hf])
    latent_matrix = np.vstack([agg[n]["last_latent"] for n in valid_hf])
    with c1:
        st.plotly_chart(
            heatmap_chart(np.round(cosine_distances(emb_matrix), 4), valid_hf, "Embedding cosine distance", "Blues"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            heatmap_chart(np.round(cosine_distances(latent_matrix), 4), valid_hf, "Latent state cosine distance", "Purples"),
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Full Summary Table</div>', unsafe_allow_html=True)

rows = []
for name in selected_models:
    a   = agg[name]
    ed  = safe_mean(a["emb_drifts"])
    ld  = safe_mean(a["latent_drifts"])
    cp  = safe_mean(a["clean_perp_list"])
    dp  = safe_mean(a["perp_list"])
    acc = safe_mean(a["accuracies"])
    wds = model_scores.get(name, float("nan"))
    rows.append({
        "Model":              name,
        "Size":               SLM_REGISTRY[name]["size"],
        "Input mode":         input_mode,
        "Benchmark":          selected_benchmark if selected_benchmark else ("Prompt drift" if prompt_drift_mode else "Manual"),
        "Task group":         bench_cfg["group"]     if bench_cfg else ("Prompt drift" if prompt_drift_mode else "—"),
        "Samples":            len(a["hall_list"]),
        "Drift type":         ("Cross-domain prompt" if prompt_drift_mode else f"{drift_level} · {DRIFT_TYPES[drift_level]}"),
        "Emb. drift (L2)":    round(ed, 4) if not np.isnan(ed) else "—",
        "Latent drift (L2)":  round(ld, 4) if not np.isnan(ld) else "—",
        "Clean perplexity":   round(cp, 2) if not np.isnan(cp) else "—",
        "Drifted perplexity": round(dp, 2) if not np.isnan(dp) else "—",
        "Δ perplexity":       round(dp - cp, 2) if (not np.isnan(dp) and not np.isnan(cp)) else "—",
        "Hall. score":        round(safe_mean(a["hall_list"]), 4),
        "Task accuracy":      f"{acc:.1%}" if not np.isnan(acc) else "N/A",
        "Weighted score":     round(wds, 4) if not np.isnan(wds) else "—",
        "Selected":           "★ YES" if name == best_model else "—",
        "Status":             "⚠ HIGH" if safe_mean(a["hall_list"]) > drift_threshold else "✓ OK",
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────
if selected_models:
    st.markdown('<div class="section-header">Multi-Metric Radar</div>', unsafe_allow_html=True)

    def norm01(vals):
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if not clean: return [0.0] * len(vals)
        mn, mx = min(clean), max(clean)
        if mx == mn: return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) if not (isinstance(v, float) and np.isnan(v)) else 0 for v in vals]

    categories = ["Perplexity", "Hallucination", "Conf. (inv)", "Emb. drift", "Accuracy (inv)"]
    raw = {
        "Perplexity":      [safe_mean(agg[n]["perp_list"]) for n in selected_models],
        "Hallucination":   [safe_mean(agg[n]["hall_list"]) for n in selected_models],
        "Conf. (inv)":     [1 - safe_mean(agg[n]["conf_list"]) if not np.isnan(safe_mean(agg[n]["conf_list"])) else 0.5 for n in selected_models],
        "Emb. drift":      [safe_mean(agg[n]["emb_drifts"]) for n in selected_models],
        "Accuracy (inv)":  [1 - safe_mean(agg[n]["accuracies"]) if not np.isnan(safe_mean(agg[n]["accuracies"])) else 0.5 for n in selected_models],
    }

    fig_radar = go.Figure()
    for i, name in enumerate(selected_models):
        vals = [norm01(raw[k])[i] for k in categories] 
        vals += [vals[0]]
        r, g, b = (int(COLORS[i % len(COLORS)].lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            fillcolor=f"rgba({r},{g},{b},0.10)",
        ))

    fig_radar.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        polar=dict(
            bgcolor="#0d1624",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1a3050", color="#3d6080"),
            angularaxis=dict(gridcolor="#1a3050", color="#8ab4d4"),
        ),
        title=dict(text="Normalised risk radar", font=dict(size=13, color="#8ab4d4")),
        showlegend=True, height=420,
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
mode_tag = "prompt_drift" if prompt_drift_mode else (selected_benchmark or "manual")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download summary CSV",
    data=csv_data,
    file_name=f"drift_results_{mode_tag}_{drift_level}.csv",
    mime="text/csv",
)
