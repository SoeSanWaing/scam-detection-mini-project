"""
app.py — Scam Message Detector (Streamlit Cloud, CPU-only)
==========================================================
A 4-tab dashboard that serves predictions and renders pre-computed analytics
exported by the offline GPU pipeline into artifacts/.

CPU GUARDRAIL (critical for Streamlit Cloud):
    This file imports ONLY standard PyPI libraries — streamlit, torch,
    transformers, pandas, plotly, joblib, networkx, scikit-learn.
    It never imports cudf / cuml / cugraph / nemo, because Streamlit Cloud
    runs on CPU Linux instances without RAPIDS or CUDA. All GPU work happens
    offline; this app only loads the exported artifacts.

Artifacts consumed (all under artifacts/):
    bert_phishing_model/   HuggingFace BERT (config.json, model.safetensors,
                           vocab.txt) — loaded with AutoModelForSequenceClassification
    cuml_baseline.joblib   {"vectorizer": TfidfVectorizer, "model": LogisticRegression}
    graph_clusters.json    {"top_pagerank": [...], "louvain_clusters": [...]}
    model_benchmark.csv    model, precision, recall, f1, accuracy, backend

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
BERT_DIR = os.path.join(ARTIFACTS_DIR, "bert_phishing_model")
BASELINE_PATH = os.path.join(ARTIFACTS_DIR, "cuml_baseline.joblib")
GRAPH_JSON = os.path.join(ARTIFACTS_DIR, "graph_clusters.json")
BENCHMARK_CSV = os.path.join(ARTIFACTS_DIR, "model_benchmark.csv")

LABEL_NAMES = {0: "Ham / Legitimate", 1: "Scam / Phishing"}

st.set_page_config(page_title="Scam Message Detector", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Text cleaning — must match the offline pipeline (common.clean_text)
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+|\b\S+\.(?:com|net|org|io|co)/\S*", re.I)
_WS_RE = re.compile(r"\s+")
_KEEP_RE = re.compile(r"[^a-zA-Z0-9!$%./\s]")


def clean_text(text: str, lowercase: bool = False) -> str:
    """Same normalisation used when the models were trained."""
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\ufffd", " ")
    s = _URL_RE.sub(" URL ", s)
    s = _KEEP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s.lower() if lowercase else s


# --------------------------------------------------------------------------- #
# Cached loaders (load each artifact once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_bert():
    """
    Load the fine-tuned BERT in HuggingFace CPU format. Returns (tokenizer,
    model) or None if the folder is absent (e.g. BERT not yet trained on GPU).
    """
    if not os.path.isdir(BERT_DIR):
        return None
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)
        tok = AutoTokenizer.from_pretrained(BERT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
        model.eval()                      # inference mode
        torch.set_grad_enabled(False)     # no autograd on CPU inference
        return tok, model
    except Exception as exc:
        st.warning(f"BERT model present but failed to load ({exc}).")
        return None


@st.cache_resource(show_spinner=False)
def load_baseline():
    """Load the TF-IDF + LogisticRegression baseline. Returns dict or None."""
    if not os.path.exists(BASELINE_PATH):
        return None
    try:
        import joblib
        bundle = joblib.load(BASELINE_PATH)
        return bundle  # {"vectorizer": ..., "model": ...}
    except Exception as exc:
        st.warning(f"Baseline model failed to load ({exc}).")
        return None


@st.cache_data(show_spinner=False)
def load_graph():
    """Load pre-computed PageRank + Louvain analytics. Returns dict or None."""
    if not os.path.exists(GRAPH_JSON):
        return None
    with open(GRAPH_JSON, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_benchmark():
    """Load the model-comparison metrics table. Returns DataFrame or None."""
    if not os.path.exists(BENCHMARK_CSV):
        return None
    return pd.read_csv(BENCHMARK_CSV)


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def predict_bert(tok, model, text: str):
    """Return (label_id, prob_scam) for one message using BERT."""
    import torch
    enc = tok(clean_text(text, lowercase=False), truncation=True,
              padding="max_length", max_length=128, return_tensors="pt")
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    prob_scam = float(probs[1])
    return (1 if prob_scam >= 0.5 else 0), prob_scam


def predict_baseline(bundle, text: str):
    """Return (label_id, prob_scam) for one message using the TF-IDF baseline."""
    vec, clf = bundle["vectorizer"], bundle["model"]
    X = vec.transform([clean_text(text, lowercase=False)])
    # Prefer calibrated probabilities; fall back to the hard label.
    if hasattr(clf, "predict_proba"):
        prob_scam = float(clf.predict_proba(X)[0][1])
    else:
        prob_scam = float(clf.predict(X)[0])
    return (1 if prob_scam >= 0.5 else 0), prob_scam


# A tiny generic fallback, used only if graph_clusters.json is unavailable.
_FALLBACK_RISK_TOKENS = {"call", "free", "claim", "prize", "url", "txt", "win"}

# Stop words we never want to highlight even if they appear as cluster terms
# (common grammar words carry no scam signal).
_HIGHLIGHT_STOP = {
    "now", "get", "only", "please", "been", "just", "new", "here", "use",
    "next", "out", "find", "today", "day", "make", "why", "over", "still",
    "can", "do", "want", "any", "other", "more", "every", "go", "dear",
}


def build_risk_tokens(graph, min_len: int = 3):
    """
    Build the highlight vocabulary 100% from the pipeline's own results
    (artifacts/graph_clusters.json): the PageRank hub terms + every Louvain
    cluster's top terms. No hand-picked words — so highlights are traceable to
    the Module 4 analysis and consistent with the Word Network tab.
    """
    if not graph:
        return set(_FALLBACK_RISK_TOKENS)
    tokens = set()
    for item in graph.get("top_pagerank", []):          # PageRank hub terms
        t = str(item.get("term", "")).lower()
        if len(t) >= min_len and t not in _HIGHLIGHT_STOP:
            tokens.add(t)
    for cluster in graph.get("louvain_clusters", []):    # Louvain topic terms
        for term in cluster.get("top_terms", []):
            t = str(term).lower()
            if len(t) >= min_len and t not in _HIGHLIGHT_STOP:
                tokens.add(t)
    return tokens or set(_FALLBACK_RISK_TOKENS)


def highlight_risk_tokens(text: str, risk_tokens) -> str:
    """Wrap each data-derived risk token found in the text with a highlight."""
    if not risk_tokens:
        return text
    def _sub(m):
        return f"<mark style='background:#ffd6d6;border-radius:3px;'>{m.group(0)}</mark>"
    pattern = r"\b(" + "|".join(re.escape(t) for t in
                                sorted(risk_tokens, key=len, reverse=True)) + r")\b"
    return re.sub(pattern, _sub, text, flags=re.I)


def risk_banner(label_id: int, prob_scam: float):
    """Render a coloured risk banner + confidence bar."""
    if label_id == 1:
        st.markdown(
            f"<div style='padding:14px;border-radius:8px;background:#b00020;"
            f"color:white;font-weight:600;'>🚨 SCAM / PHISHING detected "
            f"({prob_scam*100:.1f}% confidence)</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='padding:14px;border-radius:8px;background:#1b7a34;"
            f"color:white;font-weight:600;'>✅ Looks legitimate "
            f"({(1-prob_scam)*100:.1f}% confidence)</div>",
            unsafe_allow_html=True)
    st.progress(min(max(prob_scam, 0.0), 1.0), text=f"Scam probability: {prob_scam*100:.1f}%")


# --------------------------------------------------------------------------- #
# Load everything once
# --------------------------------------------------------------------------- #
bert = load_bert()
baseline = load_baseline()
graph = load_graph()
benchmark = load_benchmark()

# Highlight vocabulary built from the pipeline's PageRank + Louvain results.
RISK_TOKENS = build_risk_tokens(graph)

# Build the model-selector options from what's actually available.
MODEL_OPTIONS = []
if bert is not None:
    MODEL_OPTIONS.append("BERT (fine-tuned)")
if baseline is not None:
    MODEL_OPTIONS.append("Baseline (TF-IDF + LogReg)")

st.title("🛡️ Scam Message Detection Dashboard")
st.caption("CPU-only Streamlit app serving models and analytics exported by the "
           "GPU pipeline (cuDF / cuML / BERT / cuGraph).")

# Training-scope notice: tell users what the models were trained on, so they
# know where predictions are most reliable.
st.info(
    "**What this detects:** the models were trained on a classic SMS-spam "
    "dataset, so they are strongest on **lottery / prize-claim scams, "
    "free-ringtone & mobile-content spam, voucher / reward-link scams, "
    "prize-holiday offers, and bank / KYC phishing**. Newer styles such as "
    "parcel-delivery or streaming-subscription smishing are outside the "
    "training data, so predictions on those are less reliable.",
    icon="ℹ️")

if not MODEL_OPTIONS:
    st.error(
        "No models found in `artifacts/`. Run the offline pipeline first to "
        "export `cuml_baseline.joblib` and/or `bert_phishing_model/`.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Live Detector", "📑 Batch CSV Scanner",
    "🕸️ Word Network & Topics", "📊 Model Benchmark",
])


# --------------------------------------------------------------------------- #
# Tab 1 — Live Message Detector
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Live Message Detector")

    samples = {
        "— pick a sample —": "",
        "Lottery scam": "URGENT! You have WON a 2000 cash prize! To CLAIM call 09050000337 now. FREE entry.",
        "Bank phishing": "BankOfAmerica Alert: unusual activity. Verify your account at http://bit.do/secure now.",
        "Legitimate": "Hey, are we still meeting for lunch at 1pm tomorrow? Let me know.",
    }
    col_s, col_m = st.columns([2, 1])
    with col_s:
        pick = st.selectbox("Load a sample message", list(samples.keys()))
    with col_m:
        model_choice = st.selectbox("Model", MODEL_OPTIONS or ["(none available)"])

    default_text = samples.get(pick, "")
    msg = st.text_area("Message text", value=default_text, height=140,
                       placeholder="Paste an SMS/email message here…")

    if st.button("Analyze", type="primary", disabled=not MODEL_OPTIONS):
        if not msg.strip():
            st.info("Enter a message to analyze.")
        else:
            if model_choice.startswith("BERT") and bert is not None:
                label_id, prob = predict_bert(bert[0], bert[1], msg)
            else:
                label_id, prob = predict_baseline(baseline, msg)
            risk_banner(label_id, prob)
            st.markdown("**Risk token highlights**", help="Terms common in scam messages.")
            st.markdown(
                f"<div style='padding:10px;border:1px solid #ddd;border-radius:6px;'>"
                f"{highlight_risk_tokens(msg, RISK_TOKENS)}</div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Tab 2 — Batch CSV Scanner
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Batch CSV Scanner")
    st.write("Upload a CSV with a text column; every row is classified.")

    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None and MODEL_OPTIONS:
        df = pd.read_csv(up)
        text_col = st.selectbox("Which column holds the message text?",
                                list(df.columns))
        model_choice_b = st.selectbox("Model for batch", MODEL_OPTIONS,
                                      key="batch_model")
        if st.button("Scan all rows", type="primary"):
            texts = df[text_col].astype(str).tolist()
            use_bert = model_choice_b.startswith("BERT") and bert is not None
            preds, probs = [], []
            prog = st.progress(0.0, text="Classifying…")
            for i, t in enumerate(texts):
                if use_bert:
                    lid, p = predict_bert(bert[0], bert[1], t)
                else:
                    lid, p = predict_baseline(baseline, t)
                preds.append(lid); probs.append(p)
                if i % 25 == 0 or i == len(texts) - 1:
                    prog.progress((i + 1) / len(texts), text=f"{i+1}/{len(texts)}")
            df["prediction"] = [LABEL_NAMES[p] for p in preds]
            df["scam_probability"] = [round(p, 4) for p in probs]

            counts = df["prediction"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            fig = px.pie(counts, names="label", values="count",
                         title="Classification summary",
                         color="label",
                         color_discrete_map={"Scam / Phishing": "#b00020",
                                             "Ham / Legitimate": "#1b7a34"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download labeled CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="scanned_messages.csv", mime="text/csv")
    elif up is not None:
        st.error("No model available to classify with.")


# --------------------------------------------------------------------------- #
# Tab 3 — Word Network & Topic Clusters
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Scam Word Network & Topic Clusters")
    if graph is None:
        st.info("`artifacts/graph_clusters.json` not found. Run the graph "
                "analytics step of the pipeline.")
    else:
        # --- PageRank hubs bar chart ---
        pr = pd.DataFrame(graph.get("top_pagerank", []))
        if not pr.empty:
            pr = pr.sort_values("pagerank", ascending=True)
            fig_pr = px.bar(pr, x="pagerank", y="term", orientation="h",
                            title="Top PageRank hub terms in scam messages",
                            height=600)
            st.plotly_chart(fig_pr, use_container_width=True)

        # --- Louvain clusters ---
        st.markdown(f"**Louvain communities:** {graph.get('num_communities', '?')} "
                    f"topic clusters discovered")
        clusters = graph.get("louvain_clusters", [])
        # Sort by size and show the largest.
        clusters = sorted(clusters, key=lambda c: c.get("size", 0), reverse=True)
        for c in clusters[:10]:
            with st.expander(f"Cluster {c.get('cluster_id')} — {c.get('size')} terms"):
                st.write(", ".join(c.get("top_terms", [])))


# --------------------------------------------------------------------------- #
# Tab 4 — Model Benchmark & Insights
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Model Benchmark & Insights")
    if benchmark is None:
        st.info("`artifacts/model_benchmark.csv` not found. Run the baseline "
                "and/or BERT training steps.")
    else:
        st.dataframe(benchmark, use_container_width=True)

        # Grouped bar chart of the metric columns per model.
        metric_cols = [c for c in ["precision", "recall", "f1", "accuracy"]
                       if c in benchmark.columns]
        if metric_cols and "model" in benchmark.columns:
            melted = benchmark.melt(id_vars="model", value_vars=metric_cols,
                                    var_name="metric", value_name="score")
            fig_b = px.bar(melted, x="metric", y="score", color="model",
                           barmode="group", title="Model comparison", height=450)
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown(
            "**Cased vs. uncased takeaway:** scam messages lean on capitalised "
            "cues like *URGENT*, *FREE*, *CLAIM*. A cased BERT backbone keeps "
            "that signal, so it typically edges out the uncased variant on "
            "recall for the scam class — at the cost of a slightly larger "
            "vocabulary. The TF-IDF baseline is far faster but trades away "
            "recall on subtler phishing.")

st.caption("Offline pipeline: cuDF · cuML · HuggingFace BERT · cuGraph  |  "
           "This app: CPU-only (streamlit, torch, transformers, pandas, plotly).")
