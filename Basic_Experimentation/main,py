import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
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

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1420 0%, #0a1018 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #8ab4d4 !important; }

/* Main title */
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

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1e30 0%, #0a1624 100%);
    border: 1px solid #1a3050;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
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

/* Benchmark badge */
.bench-badge {
    display: inline-block;
    border-radius: 4px;
    padding: 0.15rem 0.6rem;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin: 0 2px;
}
.bench-reasoning    { background: rgba(129,140,248,0.15); border: 1px solid #818cf8; color: #818cf8; }
.bench-understanding{ background: rgba(52,211,153,0.15);  border: 1px solid #34d399; color: #34d399; }
.bench-knowledge    { background: rgba(245,158,11,0.15);  border: 1px solid #f59e0b; color: #f59e0b; }

/* Section headers */
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

/* Response box */
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

/* Alert badges */
.badge-alert {
    display: inline-block;
    background: rgba(248, 113, 113, 0.15);
    border: 1px solid #f87171;
    color: #f87171;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
}
.badge-ok {
    display: inline-block;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid #34d399;
    color: #34d399;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
}

/* Input field */
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

/* Divider */
hr { border-color: #1a3050 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1a3050 !important;
    border-radius: 8px !important;
}

/* Info box */
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly dark theme defaults
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
# Constants & Model registry
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "DistilGPT2":   {"type": "hf", "id": "distilgpt2"},
    "GPT2":         {"type": "hf", "id": "gpt2"},
    "GPT2-Medium":  {"type": "hf", "id": "gpt2-medium"},
    "GPT2-Large":   {"type": "hf", "id": "gpt2-large"},
    "GPT-Neo-125M": {"type": "hf", "id": "EleutherAI/gpt-neo-125M"},
    "Falcon-1B":    {"type": "hf", "id": "tiiuae/falcon-rw-1b"},
    "Grok":         {"type": "grok", "id": "grok"},
}

DRIFT_TYPES = {
    0: "Clean",
    1: "Context noise",
    2: "Domain shift",
    3: "Multi-topic",
    4: "Prompt length",
    5: "Instruction style",
}

DRIFT_LABELS = DRIFT_TYPES  # backward compat alias

# ─────────────────────────────────────────────
# Benchmark registry
# Each entry defines how to load from HuggingFace
# and which column carries the prompt text / label.
# ─────────────────────────────────────────────
BENCHMARK_REGISTRY = {
    # ── Group 1: Text-based reasoning & language ──────────────────────────
    "HellaSwag": {
        "hf_path":    "Rowan/hellaswag",
        "hf_name":    None,
        "split":      "validation",
        "group":      "Reasoning",
        "group_css":  "bench-reasoning",
        "text_col":   "ctx",
        "label_col":  "label",       # index into endings list
        "choices_col":"endings",
        "task_type":  "sentence-completion-mc",
        "description":"4-way sentence completion. Tests world-knowledge grounded reasoning.",
    },
    "LAMBADA": {
        "hf_path":    "EleutherAI/lambada_openai",
        "hf_name":    None,
        "split":      "test",
        "group":      "Reasoning",
        "group_css":  "bench-reasoning",
        "text_col":   "text",
        "label_col":  None,          # last word of text is the target
        "choices_col":None,
        "task_type":  "sentence-completion",
        "description":"Long-range word prediction. Last word of passage is the answer.",
    },
    "PIQA": {
        "hf_path":    "piqa",
        "hf_name":    None,
        "split":      "validation",
        "group":      "Reasoning",
        "group_css":  "bench-reasoning",
        "text_col":   "goal",
        "label_col":  "label",
        "choices_col":"sol1,sol2",    # comma-separated column names
        "task_type":  "physical-reasoning-mc",
        "description":"Physical intuition QA. 2-choice physical commonsense.",
    },
    # ── Group 2: Language understanding (GLUE / SuperGLUE) ────────────────
    "GLUE / SST-2": {
        "hf_path":    "glue",
        "hf_name":    "sst2",
        "split":      "validation",
        "group":      "Understanding",
        "group_css":  "bench-understanding",
        "text_col":   "sentence",
        "label_col":  "label",       # 0=neg, 1=pos
        "choices_col":None,
        "task_type":  "sentiment",
        "description":"Binary sentiment classification from movie reviews.",
    },
    "GLUE / MNLI": {
        "hf_path":    "glue",
        "hf_name":    "mnli",
        "split":      "validation_matched",
        "group":      "Understanding",
        "group_css":  "bench-understanding",
        "text_col":   "premise",
        "label_col":  "label",       # 0=entailment,1=neutral,2=contradiction
        "choices_col":None,
        "task_type":  "nli",
        "description":"Natural language inference: premise + hypothesis pairs.",
    },
    "SuperGLUE / BoolQ": {
        "hf_path":    "super_glue",
        "hf_name":    "boolq",
        "split":      "validation",
        "group":      "Understanding",
        "group_css":  "bench-understanding",
        "text_col":   "question",
        "label_col":  "label",       # 0=false, 1=true
        "choices_col":None,
        "task_type":  "boolean-qa",
        "description":"Yes/no QA with a reading-comprehension passage.",
    },
    "SuperGLUE / CB": {
        "hf_path":    "super_glue",
        "hf_name":    "cb",
        "split":      "validation",
        "group":      "Understanding",
        "group_css":  "bench-understanding",
        "text_col":   "premise",
        "label_col":  "label",
        "choices_col":None,
        "task_type":  "nli",
        "description":"Commitment bank: 3-class textual entailment.",
    },
    # ── Group 3: Knowledge benchmark (MMLU) ───────────────────────────────
    "MMLU / abstract_algebra": {
        "hf_path":    "cais/mmlu",
        "hf_name":    "abstract_algebra",
        "split":      "test",
        "group":      "Knowledge",
        "group_css":  "bench-knowledge",
        "text_col":   "question",
        "label_col":  "answer",      # 0-3 index into choices
        "choices_col":"choices",
        "task_type":  "knowledge-mc",
        "description":"MMLU abstract algebra subject. 4-way MCQ.",
    },
    "MMLU / anatomy": {
        "hf_path":    "cais/mmlu",
        "hf_name":    "anatomy",
        "split":      "test",
        "group":      "Knowledge",
        "group_css":  "bench-knowledge",
        "text_col":   "question",
        "label_col":  "answer",
        "choices_col":"choices",
        "task_type":  "knowledge-mc",
        "description":"MMLU anatomy subject. Tests biomedical knowledge.",
    },
    "MMLU / high_school_history": {
        "hf_path":    "cais/mmlu",
        "hf_name":    "high_school_us_history",
        "split":      "test",
        "group":      "Knowledge",
        "group_css":  "bench-knowledge",
        "text_col":   "question",
        "label_col":  "answer",
        "choices_col":"choices",
        "task_type":  "knowledge-mc",
        "description":"MMLU US history. Tests world-knowledge under domain shift.",
    },
    "MMLU / computer_security": {
        "hf_path":    "cais/mmlu",
        "hf_name":    "computer_security",
        "split":      "test",
        "group":      "Knowledge",
        "group_css":  "bench-knowledge",
        "text_col":   "question",
        "label_col":  "answer",
        "choices_col":"choices",
        "task_type":  "knowledge-mc",
        "description":"MMLU computer security subject. Closest to telecom/tech domain.",
    },
}

GROK_API_KEY = "xai-denf0c3j0T0DuWnJw6TdhghG5vuJVPP3UTcu3uJBQYDeAkLgulUKP2tuLcDZcn51EGYBJ9WWUyT4fzF1"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Configuration")

    SLM_MODELS = ["DistilGPT2", "GPT2"]
    selected_models = st.multiselect(
        "Models",
        SLM_MODELS,
        default=["DistilGPT2"],
    )

    drift_type_options = [(k, v) for k, v in DRIFT_TYPES.items()]
    drift_level = st.select_slider(
        "Drift injection type",
        options=[d[0] for d in drift_type_options],
        format_func=lambda x: f"{x} · {DRIFT_TYPES[x]}",
        value=0,
    )
    max_gen_tokens = st.slider("Max generation tokens", 50, 300, 150, step=25)

    st.divider()
    st.markdown("**Device:** `" + DEVICE + "`")
    st.markdown("**Models loaded:** cached via `@st.cache_resource`")

    # ── Dataset / Benchmark Panel ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Dataset Input Panel")

    input_mode = st.radio(
        "Input mode",
        ["Benchmark dataset", "Manual prompt"],
        index=0,
        horizontal=True,
    )

    if input_mode == "Benchmark dataset":
        # Group benchmarks for display
        bench_names = list(BENCHMARK_REGISTRY.keys())
        selected_benchmark = st.selectbox(
            "Benchmark",
            bench_names,
            format_func=lambda x: f"{x}  [{BENCHMARK_REGISTRY[x]['group']}]",
        )
        bench_cfg = BENCHMARK_REGISTRY[selected_benchmark]
        num_samples = st.slider("Samples per run", 5, 50, 15, step=5)

        group_color = {
            "Reasoning":     "#818cf8",
            "Understanding": "#34d399",
            "Knowledge":     "#f59e0b",
        }.get(bench_cfg["group"], "#8ab4d4")

        st.markdown(
            f"<div style='font-size:0.72rem;margin-top:0.4rem;'>"
            f"<span style='color:{group_color};font-weight:600;'>{bench_cfg['group']}</span>"
            f" &nbsp;·&nbsp; <span style='color:#5a7a94;'>{bench_cfg['task_type']}</span><br>"
            f"<span style='color:#3d6080;'>{bench_cfg['description']}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        selected_benchmark = None
        bench_cfg = None
        num_samples = 1

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">Neural Drift Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">LLM embedding drift · hallucination scoring · latent analysis · benchmark evaluation</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


loaded_models = {}
for name in selected_models:
    info = MODELS[name]
    if info["type"] == "hf":
        with st.spinner(f"Loading {name}…"):
            loaded_models[name] = load_model(info["id"])
    else:
        loaded_models[name] = None

# ─────────────────────────────────────────────
# Benchmark loader
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_benchmark_samples(benchmark_name: str, n: int) -> list[dict]:
    """
    Load n samples from a HuggingFace benchmark dataset.
    Returns list of dicts with keys: text, label, choices (optional).
    For MCQ datasets the choices are appended to the text as numbered options.
    """
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

        # Build choices list
        choices = None
        if cfg["choices_col"]:
            if "," in cfg["choices_col"]:
                # Multi-column choices (e.g. PIQA sol1, sol2)
                cols = [c.strip() for c in cfg["choices_col"].split(",")]
                choices = [str(row.get(c, "")) for c in cols]
            else:
                choices = row.get(cfg["choices_col"], [])
                if not isinstance(choices, list):
                    choices = list(choices)

        # Append choices to prompt text for MCQ tasks
        if choices:
            options_str = "\n".join(f"({i}) {c}" for i, c in enumerate(choices))
            prompt_text = f"{raw_text}\n{options_str}"
        else:
            prompt_text = raw_text

        samples.append({
            "text":    prompt_text,
            "raw":     raw_text,
            "label":   label,
            "choices": choices,
        })

    return samples


def get_correct_text(sample: dict, cfg: dict) -> str:
    """Return the correct answer string for accuracy computation."""
    if sample["label"] is None or sample["choices"] is None:
        return ""
    try:
        idx = int(sample["label"])
        return sample["choices"][idx]
    except Exception:
        return str(sample["label"])


def compute_accuracy(generated: str, correct: str) -> float:
    """Simple token-overlap accuracy for MCQ tasks."""
    if not correct:
        return float("nan")
    gen_tokens = set(generated.lower().split())
    cor_tokens = set(correct.lower().split())
    if not cor_tokens:
        return 0.0
    overlap = gen_tokens & cor_tokens
    return len(overlap) / len(cor_tokens)


# ─────────────────────────────────────────────
# Drift injection
# ─────────────────────────────────────────────
def inject_drift(prompt: str, level: int) -> str:
    if level == 0:
        return prompt
    elif level == 1:
        return prompt + " Also consider unrelated context about streaming services."
    elif level == 2:
        return prompt + " Also discuss Roman law history."
    elif level == 3:
        return prompt + " Also summarize climate change policies."
    elif level == 4:
        return (prompt + " ") * 30 + "Please provide a detailed answer."
    elif level == 5:
        return "Write a 500 word detailed research explanation about: " + prompt + " Include examples and references."
    return prompt


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────
def extract_features(tokenizer, model, prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.hidden_states[-1]
    embedding   = last_hidden.mean(dim=1).cpu().numpy()
    latent      = [h.mean(dim=1).cpu().numpy() for h in outputs.hidden_states]
    logits      = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    perplexity = torch.exp(loss).item()
    probs      = torch.softmax(logits, dim=-1)
    confidence = torch.max(probs, dim=-1)[0].mean().item()
    return embedding, latent, perplexity, confidence, logits


def grok_generate(prompt: str, max_tokens: int = 150) -> str:
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": "grok-4-latest",
        "messages": [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user",   "content": prompt},
        ],
        "stream": False, "temperature": 0, "max_tokens": max_tokens,
    }
    r = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=30)
    return (
        r.json()["choices"][0]["message"]["content"]
        if r.status_code == 200
        else f"[Grok Error {r.status_code}]"
    )


def hall_score(confidence, perplexity, response_len):
    if not np.isnan(confidence):
        return (
            (1 - confidence) * 0.5
            + (perplexity / 100) * 0.3
            + (response_len / 200) * 0.2
        )
    return (response_len / 200) * 0.2


# ─────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────
def bar_chart(names, values, title, color_idx=0, unit=""):
    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(
            color=values,
            colorscale=[
                [0,   "#1a3050"],
                [0.5, COLORS[color_idx]],
                [1.0, COLORS[(color_idx + 1) % len(COLORS)]],
            ],
            line=dict(color="#080c14", width=1),
        ),
        text=[f"{v:.3f}" if v else "N/A" for v in values],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#8ab4d4")),
        yaxis_title=unit,
        showlegend=False,
        height=300,
    )
    return fig


def heatmap_chart(matrix, labels, title, colorscale="Blues"):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale=colorscale,
        text=np.round(matrix, 4),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#8ab4d4")),
        height=320,
    )
    return fig


def layer_drift_chart(drift_per_layer, name):
    layers = list(range(len(drift_per_layer)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=drift_per_layer,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=6, color="#818cf8"),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.08)",
    ))
    fig.add_trace(go.Bar(
        x=layers, y=drift_per_layer,
        marker=dict(
            color=drift_per_layer,
            colorscale=[[0, "#1a3050"], [1, "#f87171"]],
            opacity=0.35,
        ),
        showlegend=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"Layer-wise drift · {name}", font=dict(size=13, color="#8ab4d4")),
        xaxis_title="Layer",
        yaxis_title="L2 drift",
        height=250,
    )
    return fig


def benchmark_comparison_chart(bench_results: list[dict]):
    """
    Grouped bar chart: one group per benchmark,
    bars = perplexity delta, emb drift, hall score.
    bench_results: list of dicts with keys benchmark, model,
                   perp_delta, emb_drift, hall_score.
    """
    df = pd.DataFrame(bench_results)
    if df.empty:
        return None

    benchmarks = df["benchmark"].tolist()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Perplexity Δ",
        x=benchmarks,
        y=df["perp_delta"].tolist(),
        marker_color="#38bdf8",
        text=[f"{v:.2f}" for v in df["perp_delta"]],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig.add_trace(go.Bar(
        name="Emb. drift (L2)",
        x=benchmarks,
        y=df["emb_drift"].tolist(),
        marker_color="#818cf8",
        text=[f"{v:.4f}" for v in df["emb_drift"]],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig.add_trace(go.Bar(
        name="Hall. score ×10",
        x=benchmarks,
        y=[v * 10 for v in df["hall_score"]],
        marker_color="#f87171",
        text=[f"{v:.3f}" for v in df["hall_score"]],
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Drift metrics comparison across benchmarks",
            font=dict(size=13, color="#8ab4d4"),
        ),
        barmode="group",
        height=360,
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(tickangle=-25, gridcolor="#1a3050"),
    )
    return fig


# ─────────────────────────────────────────────
# Build prompt list
# ─────────────────────────────────────────────
if input_mode == "Benchmark dataset":
    st.markdown(
        f'<div class="info-box">'
        f'<b style="color:#38bdf8">{selected_benchmark}</b>'
        f' &nbsp;[{bench_cfg["group"]}]&nbsp; — {bench_cfg["description"]}<br>'
        f'Loading <b>{num_samples}</b> samples · task type: <code>{bench_cfg["task_type"]}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.spinner(f"Loading {selected_benchmark} from HuggingFace…"):
        samples = load_benchmark_samples(selected_benchmark, num_samples)

    if not samples:
        st.error("No samples loaded. Check dataset name / HuggingFace connectivity.")
        st.stop()

    # Optional single-prompt override
    manual_override = st.text_input(
        "Override with a single custom prompt (leave blank to run all benchmark samples)",
        placeholder="Optional — leave blank to use benchmark…",
        label_visibility="visible",
    )
    if manual_override.strip():
        prompts_to_run = [{"text": manual_override, "raw": manual_override, "label": None, "choices": None}]
    else:
        prompts_to_run = samples

else:
    # Manual prompt mode
    manual_prompt = st.text_input(
        "Prompt",
        placeholder="Enter a test prompt…",
        label_visibility="collapsed",
    )
    if not manual_prompt or not selected_models:
        st.markdown(
            "<div style='color:#3d6080;text-align:center;padding:4rem 0;"
            "font-size:0.85rem;letter-spacing:0.1em;'>"
            "SELECT MODELS IN THE SIDEBAR · THEN ENTER A PROMPT ABOVE"
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()
    prompts_to_run = [{"text": manual_prompt, "raw": manual_prompt, "label": None, "choices": None}]

if not selected_models:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────
# Show drift-modified example
# ─────────────────────────────────────────────
if drift_level > 0:
    example_modified = inject_drift(prompts_to_run[0]["text"], drift_level)
    with st.expander(f"🔀 Example modified prompt — drift level {drift_level} ({DRIFT_TYPES[drift_level]})", expanded=False):
        st.code(example_modified[:800] + ("…" if len(example_modified) > 800 else ""))

# ─────────────────────────────────────────────
# Run inference across all samples × models
# ─────────────────────────────────────────────
# Aggregated results per model
agg = {
    name: {
        "perp_list":    [],
        "clean_perp_list": [],
        "emb_drifts":   [],
        "latent_drifts":[],
        "hall_list":    [],
        "conf_list":    [],
        "resp_lens":    [],
        "accuracies":   [],
        "responses":    [],
        # for cross-model similarity — store last sample
        "last_emb":     None,
        "last_latent":  None,
        "last_clean_emb":    None,
        "last_clean_latent": None,
        "last_latent_layers": None,
    }
    for name in selected_models
}

progress_bar = st.progress(0, text="Running inference…")
total_steps  = len(prompts_to_run) * len(selected_models)
step         = 0

for sample in prompts_to_run:
    raw_text      = sample["text"]
    modified_text = inject_drift(raw_text, drift_level)
    correct_text  = get_correct_text(sample, bench_cfg) if bench_cfg else ""

    for name in selected_models:
        info = MODELS[name]
        step += 1
        progress_bar.progress(step / total_steps, text=f"Running {name} on sample {step}/{total_steps}…")

        if info["type"] == "hf":
            tokenizer, model = loaded_models[name]

            # Generate response
            inputs = tokenizer(
                modified_text, return_tensors="pt", truncation=True, max_length=512
            ).to(DEVICE)
            with torch.no_grad():
                gen = model.generate(inputs["input_ids"], max_new_tokens=max_gen_tokens)
            response = tokenizer.decode(gen[0], skip_special_tokens=True)

            # Drifted features
            emb, latent, perp, conf, _ = extract_features(tokenizer, model, modified_text)
            # Clean baseline features
            c_emb, c_latent, c_perp, _, _ = extract_features(tokenizer, model, raw_text)

            emb_flat    = emb.flatten()
            c_emb_flat  = c_emb.flatten()
            lat_flat    = np.mean([l.flatten() for l in latent], axis=0)
            c_lat_flat  = np.mean([l.flatten() for l in c_latent], axis=0)

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
            # Keep last sample for per-layer and heatmap analysis
            a["last_emb"]           = emb_flat
            a["last_latent"]        = lat_flat
            a["last_clean_emb"]     = c_emb_flat
            a["last_clean_latent"]  = c_lat_flat
            a["last_latent_layers"] = [l.flatten() for l in latent]

        else:
            response = grok_generate(modified_text, max_gen_tokens)
            acc      = compute_accuracy(response, correct_text)
            hall     = hall_score(float("nan"), 0, len(response.split()))
            a = agg[name]
            a["hall_list"].append(hall)
            a["conf_list"].append(float("nan"))
            a["resp_lens"].append(len(response.split()))
            a["accuracies"].append(acc)
            a["responses"].append(response)

progress_bar.empty()


def safe_mean(lst):
    vals = [v for v in lst if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">At-a-Glance Metrics</div>', unsafe_allow_html=True)

bench_label_html = ""
if bench_cfg:
    css  = bench_cfg["group_css"]
    grp  = bench_cfg["group"]
    task = bench_cfg["task_type"]
    bench_label_html = (
        f'<span class="bench-badge {css}">{selected_benchmark}</span>'
        f'<span class="bench-badge" style="background:rgba(56,189,248,0.1);'
        f'border:1px solid #38bdf8;color:#38bdf8;">{task}</span>'
    )

kpi_cols = st.columns(len(selected_models))
for col, name in zip(kpi_cols, selected_models):
    a     = agg[name]
    h     = safe_mean(a["hall_list"])
    p     = safe_mean(a["perp_list"])
    c     = safe_mean(a["conf_list"])
    ed    = safe_mean(a["emb_drifts"])
    acc   = safe_mean(a["accuracies"])
    alert = h > 0.5
    badge = '<span class="badge-alert">⚠ HIGH RISK</span>' if alert else '<span class="badge-ok">✓ NOMINAL</span>'
    p_str   = f"{p:.1f}"   if not np.isnan(p)   else "N/A"
    c_str   = f"{c:.3f}"   if not np.isnan(c)   else "N/A"
    ed_str  = f"{ed:.4f}"  if not np.isnan(ed)  else "N/A"
    acc_str = f"{acc:.1%}" if not np.isnan(acc) else "N/A"
    n_str   = str(len(a["hall_list"]))

    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size:0.7rem;letter-spacing:0.12em;color:#3d6080;text-transform:uppercase;margin-bottom:0.4rem;'>{name}</div>
            <div class="metric-value {'metric-alert' if alert else 'metric-ok'}">{h:.3f}</div>
            <div class="metric-label">Hallucination Score (mean)</div>
            <div style='margin-top:0.8rem;'>{badge}</div>
            <div style='margin-top:0.8rem;font-size:0.7rem;color:#3d6080;line-height:1.8;'>
                Perplexity &nbsp;<b style='color:#8ab4d4;'>{p_str}</b> &nbsp;|&nbsp;
                Conf &nbsp;<b style='color:#8ab4d4;'>{c_str}</b><br>
                Emb drift &nbsp;<b style='color:#8ab4d4;'>{ed_str}</b> &nbsp;|&nbsp;
                Samples &nbsp;<b style='color:#8ab4d4;'>{n_str}</b><br>
                Task accuracy &nbsp;<b style='color:#{"34d399" if not np.isnan(acc) and acc > 0.5 else "f87171"};'>{acc_str}</b>
            </div>
            <div style='margin-top:0.6rem;'>{bench_label_html}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sample responses
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Model Responses (last sample)</div>', unsafe_allow_html=True)
resp_cols = st.columns(min(len(selected_models), 3))
for i, name in enumerate(selected_models):
    a = agg[name]
    last_resp = a["responses"][-1] if a["responses"] else "—"
    with resp_cols[i % 3]:
        st.markdown(f"**{name}**")
        st.markdown(f'<div class="response-box">{last_resp}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Core metrics charts
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Core Metrics</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

names = selected_models
perps  = [safe_mean(agg[n]["perp_list"]) for n in names]
halls  = [safe_mean(agg[n]["hall_list"]) for n in names]
confs  = [safe_mean(agg[n]["conf_list"]) for n in names]
rlens  = [safe_mean(agg[n]["resp_lens"]) for n in names]
accs   = [safe_mean(agg[n]["accuracies"]) for n in names]

with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Perplexity", x=names, y=perps, marker_color="#38bdf8",
        text=[f"{v:.1f}" for v in perps], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Hall. score ×100", x=names, y=[h * 100 for h in halls],
        marker_color="#f87171",
        text=[f"{v:.1f}" for v in [h * 100 for h in halls]], textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Perplexity vs hallucination score", font=dict(size=13, color="#8ab4d4")),
        barmode="group", height=320,
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Confidence", x=names, y=confs, marker_color="#34d399",
        text=[f"{v:.3f}" for v in confs], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Task accuracy", x=names, y=accs, marker_color="#f59e0b",
        text=[f"{v:.1%}" if not np.isnan(v) else "N/A" for v in accs],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Confidence & task accuracy", font=dict(size=13, color="#8ab4d4")),
        barmode="group", height=320,
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# Drift analysis (only when drift > 0)
# ─────────────────────────────────────────────
hf_names = [n for n in names if MODELS[n]["type"] == "hf" and agg[n]["last_emb"] is not None]

if drift_level > 0 and hf_names:
    st.markdown(
        f'<div class="section-header">Drift Analysis · Level {drift_level} — {DRIFT_TYPES[drift_level]}</div>',
        unsafe_allow_html=True,
    )

    emb_drifts    = [safe_mean(agg[n]["emb_drifts"])    for n in hf_names]
    latent_drifts = [safe_mean(agg[n]["latent_drifts"]) for n in hf_names]
    perp_deltas   = [
        safe_mean(agg[n]["perp_list"]) - safe_mean(agg[n]["clean_perp_list"])
        for n in hf_names
    ]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(bar_chart(hf_names, emb_drifts,  "Embedding drift (L2)",   0, "L2 norm"),  use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(hf_names, latent_drifts,"Latent state drift (L2)",2, "L2 norm"),  use_container_width=True)
    with c3:
        st.plotly_chart(bar_chart(hf_names, perp_deltas, "Perplexity Δ vs clean",  4, "Δ pplx"),   use_container_width=True)

    # Layer-wise drift
    st.markdown('<div class="section-header">Layer-wise Drift</div>', unsafe_allow_html=True)
    layer_cols = st.columns(min(len(hf_names), 3))
    for i, name in enumerate(hf_names):
        tokenizer, model = loaded_models[name]
        _, c_lat, _, _, _ = extract_features(tokenizer, model, prompts_to_run[0]["text"])
        clean_layers = [l.flatten() for l in c_lat]
        drifts = [
            float(np.linalg.norm(agg[name]["last_latent_layers"][j] - clean_layers[j]))
            for j in range(len(clean_layers))
        ]
        with layer_cols[i % 3]:
            st.plotly_chart(layer_drift_chart(drifts, name), use_container_width=True)

# ─────────────────────────────────────────────
# Benchmark-level drift comparison chart
# (shown only when running benchmark mode with drift)
# ─────────────────────────────────────────────
if input_mode == "Benchmark dataset" and drift_level > 0 and hf_names:
    st.markdown('<div class="section-header">Benchmark Drift Profile</div>', unsafe_allow_html=True)
    bench_results = []
    for name in hf_names:
        bench_results.append({
            "benchmark":  selected_benchmark,
            "model":      name,
            "perp_delta": safe_mean(agg[name]["perp_list"]) - safe_mean(agg[name]["clean_perp_list"]),
            "emb_drift":  safe_mean(agg[name]["emb_drifts"]),
            "hall_score": safe_mean(agg[name]["hall_list"]),
        })
    fig_bench = benchmark_comparison_chart(bench_results)
    if fig_bench:
        st.plotly_chart(fig_bench, use_container_width=True)

    # Per-sample perplexity drift sparkline
    st.markdown('<div class="section-header">Per-sample Perplexity over Run</div>', unsafe_allow_html=True)
    spark_cols = st.columns(min(len(hf_names), 3))
    for i, name in enumerate(hf_names):
        a = agg[name]
        if a["perp_list"] and a["clean_perp_list"]:
            deltas = [p - c for p, c in zip(a["perp_list"], a["clean_perp_list"])]
            fig_spark = go.Figure()
            fig_spark.add_trace(go.Scatter(
                x=list(range(len(deltas))),
                y=deltas,
                mode="lines+markers",
                line=dict(color="#38bdf8", width=1.5),
                marker=dict(size=4, color="#818cf8"),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.07)",
                name="Δ perplexity",
            ))
            fig_spark.add_hline(y=0, line_color="#3d6080", line_width=1, line_dash="dash")
            fig_spark.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text=f"Δ perplexity per sample · {name}", font=dict(size=12, color="#8ab4d4")),
                xaxis_title="Sample index",
                yaxis_title="Δ pplx",
                height=220,
                margin=dict(l=40, r=10, t=40, b=30),
            )
            with spark_cols[i % 3]:
                st.plotly_chart(fig_spark, use_container_width=True)

# ─────────────────────────────────────────────
# Cross-model similarity heatmaps
# ─────────────────────────────────────────────
valid_hf = [n for n in hf_names if agg[n]["last_emb"] is not None]
if len(valid_hf) > 1:
    st.markdown('<div class="section-header">Cross-Model Similarity</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    emb_matrix    = np.vstack([agg[n]["last_emb"]    for n in valid_hf])
    latent_matrix = np.vstack([agg[n]["last_latent"] for n in valid_hf])
    emb_dist      = cosine_distances(emb_matrix)
    latent_dist   = cosine_distances(latent_matrix)

    with c1:
        st.plotly_chart(
            heatmap_chart(np.round(emb_dist, 4), valid_hf, "Embedding cosine distance", "Blues"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            heatmap_chart(np.round(latent_dist, 4), valid_hf, "Latent state cosine distance", "Purples"),
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# Summary dataframe
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Full Summary Table</div>', unsafe_allow_html=True)

rows = []
for name in selected_models:
    a = agg[name]
    ed  = safe_mean(a["emb_drifts"])
    ld  = safe_mean(a["latent_drifts"])
    cp  = safe_mean(a["clean_perp_list"])
    dp  = safe_mean(a["perp_list"])
    acc = safe_mean(a["accuracies"])
    rows.append({
        "Model":              name,
        "Benchmark":          selected_benchmark if selected_benchmark else "Manual prompt",
        "Task group":         bench_cfg["group"]     if bench_cfg else "—",
        "Task type":          bench_cfg["task_type"] if bench_cfg else "—",
        "Samples":            len(a["hall_list"]),
        "Drift type":         f"{drift_level} · {DRIFT_TYPES[drift_level]}",
        "Emb. drift (L2)":    round(ed, 4)  if not np.isnan(ed) else "—",
        "Latent drift (L2)":  round(ld, 4)  if not np.isnan(ld) else "—",
        "Clean perplexity":   round(cp, 2)  if not np.isnan(cp) else "—",
        "Drifted perplexity": round(dp, 2)  if not np.isnan(dp) else "—",
        "Δ perplexity":       round(dp - cp, 2) if (not np.isnan(dp) and not np.isnan(cp)) else "—",
        "Confidence":         round(safe_mean(a["conf_list"]), 4),
        "Hall. score":        round(safe_mean(a["hall_list"]), 4),
        "Task accuracy":      f"{acc:.1%}" if not np.isnan(acc) else "N/A",
        "Status":             "⚠ HIGH" if safe_mean(a["hall_list"]) > 0.5 else "✓ OK",
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Radar chart: multi-metric per model
# ─────────────────────────────────────────────
if selected_models:
    st.markdown('<div class="section-header">Multi-Metric Radar</div>', unsafe_allow_html=True)

    def norm01(vals):
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if not clean:
            return [0.0] * len(vals)
        mn, mx = min(clean), max(clean)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) if not (isinstance(v, float) and np.isnan(v)) else 0 for v in vals]

    categories = ["Perplexity", "Hallucination", "Confidence (inv)", "Emb. drift", "Task accuracy (inv)"]
    fig_radar  = go.Figure()

    raw = {
        "Perplexity":          [safe_mean(agg[n]["perp_list"])  for n in selected_models],
        "Hallucination":       [safe_mean(agg[n]["hall_list"])  for n in selected_models],
        "Confidence (inv)":    [1 - safe_mean(agg[n]["conf_list"]) if not np.isnan(safe_mean(agg[n]["conf_list"])) else 0.5 for n in selected_models],
        "Emb. drift":          [safe_mean(agg[n]["emb_drifts"]) for n in selected_models],
        "Task accuracy (inv)": [1 - safe_mean(agg[n]["accuracies"]) if not np.isnan(safe_mean(agg[n]["accuracies"])) else 0.5 for n in selected_models],
    }

    for i, name in enumerate(selected_models):
        vals = [norm01(raw[k])[i] for k in categories]
        vals += [vals[0]]
        r, g, b = (
            int(COLORS[i % len(COLORS)].lstrip("#")[0:2], 16),
            int(COLORS[i % len(COLORS)].lstrip("#")[2:4], 16),
            int(COLORS[i % len(COLORS)].lstrip("#")[4:6], 16),
        )
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
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
        showlegend=True,
        height=420,
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# Export results as CSV
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download summary CSV",
    data=csv_data,
    file_name=f"drift_results_{selected_benchmark or 'manual'}_{drift_level}.csv",
    mime="text/csv",
)
