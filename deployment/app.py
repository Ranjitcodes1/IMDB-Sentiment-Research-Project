import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import streamlit.components.v1 as components

# Page config
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #e50914;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stTextArea textarea {
        background-color: #1a1c23;
        color: white;
        border-radius: 8px;
        border: 1px solid #333;
    }
    .stButton button {
        width: 100%;
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #f40612;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    import os
    
    # Get absolute path to the project root assuming this script is in deployment/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "models", "distilbert_imdb", "distilbert_imdb")
    
    try:
        # Load local tokenizer or fallback to base
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

@st.cache_resource
def get_explainer(_model, _tokenizer):
    if _model is None or _tokenizer is None:
        return None
    # Use pipeline for SHAP
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=_model, tokenizer=_tokenizer, device=device)
    explainer = shap.Explainer(pipe)
    return explainer

explainer = get_explainer(model, tokenizer)

# Header
st.markdown("<h1>🎬 IMDb Movie Review Sentiment</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Powered by a fine-tuned DistilBERT transformer model.</p>", unsafe_allow_html=True)

# Input
review_text = st.text_area("Write your movie review here...", height=200, placeholder="e.g., I absolutely loved this movie! The acting was phenomenal and the plot was engaging from start to finish.")

if st.button("Analyze Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review to analyze.")
    elif model is None:
        st.error("Model is not loaded properly.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize input
            inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                
            # DistilBERT outputs 2 classes: 0 -> Negative, 1 -> Positive
            neg_prob = probs[0][0].item()
            pos_prob = probs[0][1].item()
            
            sentiment = "Positive" if pos_prob > neg_prob else "Negative"
            confidence = max(pos_prob, neg_prob)
            
            # Display results
            st.markdown("---")
            if sentiment == "Positive":
                st.success(f"**Sentiment:** Positive 🎥")
            else:
                st.error(f"**Sentiment:** Negative 📉")
                
            st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
            
            # Progress bar for probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Positive Probability", value=f"{pos_prob * 100:.1f}%")
                st.progress(pos_prob)
            with col2:
                st.metric(label="Negative Probability", value=f"{neg_prob * 100:.1f}%")
                st.progress(neg_prob)
                
            st.markdown("### Model Explainability")
            st.markdown("See which words influenced the prediction. Words in **red** drove the prediction toward Positive, and words in **blue** drove it toward Negative.")
            
            with st.spinner("Generating SHAP explanation (this may take a moment)..."):
                try:
                    shap_values = explainer([review_text])
                    # shap_values.values has shape (num_samples, num_tokens, num_classes)
                    # We want the explanation for the predicted class
                    # For DistilBERT, label 1 is usually Positive
                    pred_class_idx = 1 if sentiment == "Positive" else 0
                    
                    # Create HTML for the text plot
                    html = shap.plots.text(shap_values[0, :, pred_class_idx], display=False)
                    components.html(html, height=300, scrolling=True)
                except Exception as e:
                    st.error(f"Could not generate SHAP explanation: {e}")
