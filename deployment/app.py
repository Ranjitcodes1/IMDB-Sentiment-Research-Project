import os
import streamlit as st
import torch
import torch.nn.functional as F
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import streamlit.components.v1 as components

# ---------------------------------------------------------
# Page Configuration & Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="IMDb Sentiment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Cinematic Netflix/IMDb Theme)
st.markdown("""
<style>
    .main {
        background-color: #0b0d13;
        color: #e6e8ec;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .header-title {
        text-align: center;
        background: linear-gradient(135deg, #ff4b4b 0%, #f43f5e 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #161a23;
        border: 1px solid #232936;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .stTextArea textarea {
        background-color: #12151e !important;
        color: #f3f4f6 !important;
        border-radius: 10px !important;
        border: 1px solid #2a3142 !important;
        font-size: 15px !important;
    }
    .stTextArea textarea:focus {
        border-color: #f43f5e !important;
        box-shadow: 0 0 0 1px #f43f5e !important;
    }
    .badge-pos {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neg {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model & Pipeline Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_baseline():
    """Loads the serialized TF-IDF + Logistic Regression pipeline."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pipeline_path = os.path.join(project_root, "models", "baseline_pipeline.pkl")
    if os.path.exists(pipeline_path):
        try:
            return joblib.load(pipeline_path)
        except Exception as e:
            st.sidebar.error(f"Error loading baseline: {e}")
    return None

@st.cache_resource
def load_distilbert():
    """Loads fine-tuned DistilBERT weights and tokenizer."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    candidate_paths = [
        os.path.join(project_root, "models", "distilbert_imdb", "distilbert_imdb"),
        os.path.join(project_root, "models", "distilbert_imdb")
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, "model.safetensors")) or os.path.exists(os.path.join(p, "pytorch_model.bin")):
            model_path = p
            break
            
    if model_path:
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            return tokenizer, model
        except Exception as e:
            st.sidebar.error(f"Error loading DistilBERT: {e}")
    return None, None

@st.cache_resource
def load_deberta():
    """Loads fine-tuned DeBERTa-v3 weights and tokenizer if present."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    candidate_paths = [
        os.path.join(project_root, "models", "deberta_v3_imdb"),
        os.path.join(project_root, "models", "deberta_v3_imdb", "deberta_v3_imdb")
    ]
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, "model.safetensors")) or os.path.exists(os.path.join(p, "pytorch_model.bin")):
            try:
                tokenizer = AutoTokenizer.from_pretrained(p)
                model = AutoModelForSequenceClassification.from_pretrained(p)
                model.eval()
                return tokenizer, model
            except Exception:
                pass
    return None, None

@st.cache_resource
def get_explainer(_model, _tokenizer):
    """Initializes SHAP text classification pipeline explainer."""
    if _model is None or _tokenizer is None:
        return None
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-classification", model=_model, tokenizer=_tokenizer, device=device)
        return shap.Explainer(pipe)
    except Exception:
        return None

baseline_pipe = load_baseline()
tokenizer, distilbert_model = load_distilbert()
deberta_tokenizer, deberta_model = load_deberta()

# ---------------------------------------------------------
# Sidebar Controls & Information
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=90)
    st.markdown("### Model Configuration")
    
    engine_options = ["Compare Both Models Side-by-Side"]
    if deberta_model:
        engine_options.append("DeBERTa-v3 (SOTA)")
    engine_options.extend(["DistilBERT (Transformer)", "Optimized TF-IDF Baseline"])
    
    model_choice = st.radio(
        "Select Sentiment Engine:",
        engine_options,
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Benchmark Accuracy")
    st.markdown("""
    - **DistilBERT**: `89.74%`
    - **TF-IDF Baseline**: `88.98%`
    - **DeBERTa-v3 (Target)**: `95.50%+`
    """)
    
    enable_shap = st.checkbox(
        "Enable SHAP Token Attribution",
        value=False,
        help="Visualizes token contributions. May add a brief inference latency on CPU."
    )
    
    st.markdown("---")
    st.caption("Built with PyTorch, Transformers, Scikit-learn, and Streamlit.")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("<div class='header-title'>IMDb Sentiment Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>State-of-the-Art NLP comparing classical and transformer architectures.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Review Buttons
# ---------------------------------------------------------
st.markdown("##### ⚡ Quick Test Presets:")

PRESETS = {
    "Sarcastic Criticism": "Oh what an absolute cinematic masterpiece of complete boredom and dreadful acting. Two hours of my life I will never get back.",
    "Mixed Nuance": "The visuals and cinematography were breathtakingly gorgeous, but unfortunately the script was utterly hollow and the dialogue felt lifeless.",
    "Enthusiastic Praise": "An absolute triumph! Remarkable acting, brilliantly directed, and kept me glued to the edge of my seat from start to finish.",
    "Decisive Negative": "Easily one of the worst movies of the decade. Disjointed, uninspired, and completely unwatchable from the opening scene."
}

if "review_input" not in st.session_state:
    st.session_state.review_input = "I absolutely loved this movie! The acting was phenomenal and the plot was engaging from start to finish."

p_cols = st.columns(len(PRESETS))
for i, (label, text) in enumerate(PRESETS.items()):
    if p_cols[i].button(label, use_container_width=True):
        st.session_state.review_input = text

# Review Text Area
review_text = st.text_area(
    "Write or paste your movie review below:",
    value=st.session_state.review_input,
    height=160
)

# Text length gauge
words = len(review_text.split())
chars = len(review_text)
length_color = "#34d399" if words <= 256 else "#fbbf24"
st.caption(f"Review stats: **{words}** words, **{chars}** characters | Transformer Context Limit: ~256 tokens")

# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        results = {}
        
        # 1. Baseline Prediction
        if baseline_pipe and model_choice in ["Compare Both Models Side-by-Side", "Optimized TF-IDF Baseline"]:
            try:
                probs = baseline_pipe.predict_proba([review_text])[0]
                pred = baseline_pipe.predict([review_text])[0]
                results["Baseline"] = {
                    "sentiment": "Positive" if pred == 1 else "Negative",
                    "pos_prob": probs[1],
                    "neg_prob": probs[0],
                    "confidence": max(probs)
                }
            except Exception as e:
                st.error(f"Baseline error: {e}")
                
        # 2. DistilBERT Prediction
        if distilbert_model and tokenizer and model_choice in ["Compare Both Models Side-by-Side", "DistilBERT (Transformer)"]:
            try:
                inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
                with torch.no_grad():
                    outputs = distilbert_model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)[0]
                    neg_prob = probs[0].item()
                    pos_prob = probs[1].item()
                    sentiment = "Positive" if pos_prob > neg_prob else "Negative"
                    results["DistilBERT"] = {
                        "sentiment": sentiment,
                        "pos_prob": pos_prob,
                        "neg_prob": neg_prob,
                        "confidence": max(pos_prob, neg_prob)
                    }
            except Exception as e:
                st.error(f"DistilBERT error: {e}")

        # Display Results
        st.markdown("---")
        
        if model_choice == "Compare Both Models Side-by-Side":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⚡ Optimized TF-IDF Baseline")
                if "Baseline" in results:
                    b_res = results["Baseline"]
                    badge = "<span class='badge-pos'>POSITIVE</span>" if b_res["sentiment"] == "Positive" else "<span class='badge-neg'>NEGATIVE</span>"
                    st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)
                    st.metric("Confidence", f"{b_res['confidence']*100:.1f}%")
                    st.write(f"Positive: **{b_res['pos_prob']*100:.1f}%** | Negative: **{b_res['neg_prob']*100:.1f}%**")
                    st.progress(b_res["pos_prob"])
                else:
                    st.info("Baseline model not loaded.")
                    
            with col2:
                st.markdown("### 🚀 DistilBERT Transformer")
                if "DistilBERT" in results:
                    d_res = results["DistilBERT"]
                    badge = "<span class='badge-pos'>POSITIVE</span>" if d_res["sentiment"] == "Positive" else "<span class='badge-neg'>NEGATIVE</span>"
                    st.markdown(f"**Verdict:** {badge}", unsafe_allow_html=True)
                    st.metric("Confidence", f"{d_res['confidence']*100:.1f}%")
                    st.write(f"Positive: **{d_res['pos_prob']*100:.1f}%** | Negative: **{d_res['neg_prob']*100:.1f}%**")
                    st.progress(d_res["pos_prob"])
                else:
                    st.info("DistilBERT model not loaded.")
                    
        else:
            active_key = "DistilBERT" if "DistilBERT" in model_choice else "Baseline"
            if active_key in results:
                res = results[active_key]
                badge = "<span class='badge-pos'>POSITIVE</span>" if res["sentiment"] == "Positive" else "<span class='badge-neg'>NEGATIVE</span>"
                st.markdown(f"### Predicted Sentiment: {badge}", unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence Score", f"{res['confidence']*100:.1f}%")
                m2.metric("Positive Probability", f"{res['pos_prob']*100:.1f}%")
                m3.metric("Negative Probability", f"{res['neg_prob']*100:.1f}%")
                st.progress(res["pos_prob"])

        # ---------------------------------------------------------
        # SHAP Explainability
        # ---------------------------------------------------------
        if enable_shap and distilbert_model and tokenizer:
            st.markdown("---")
            st.markdown("### 🔍 Model Explainability (SHAP)")
            st.caption("Tokens in **red** push the sentiment towards Positive; tokens in **blue** push towards Negative.")
            
            with st.spinner("Computing SHAP values..."):
                try:
                    explainer = get_explainer(distilbert_model, tokenizer)
                    if explainer:
                        shap_values = explainer([review_text])
                        pred_idx = 1 if results.get("DistilBERT", {}).get("sentiment") == "Positive" else 0
                        html = shap.plots.text(shap_values[0, :, pred_idx], display=False)
                        components.html(html, height=280, scrolling=True)
                    else:
                        st.info("SHAP explainer could not be initialized.")
                except Exception as e:
                    st.warning(f"Could not compute SHAP plot: {e}")
