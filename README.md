# IMDb Sentiment Analysis Research

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://imdb-sentiment-research-project-aapppp9wcxwfkurakjgjtgc.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## Overview

This research project explores binary sentiment classification on the **IMDb Large Movie Review Dataset** across the NLP modeling spectrum: from classical machine learning baselines to modern transformer architectures.

The investigation evaluates how feature engineering (negation preservation, sublinear TF-IDF, n-gram ranges), contextual attention, and sequence length handling address linguistic nuances such as sarcasm, sentiment shifts, and complex negations.

---

## Benchmark Results

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Initial TF-IDF Baseline (Unigrams)** | 87.24% | 86.80% | 87.80% | 87.30% | Completed |
| **Optimized TF-IDF Pipeline (Bigrams + Negation)** | **88.98%** | **88.11%** | **90.12%** | **89.10%** | **Completed** |
| **DistilBERT (Fine-Tuned, 256 tokens)** | **89.74%** | **87.40%** | **93.90%** | **90.00%** | **Completed** |
| **DeBERTa-v3-base (512 tokens + Head/Tail)** | *~95.50%* | *~95.20%* | *~95.80%* | *~95.50%* | Pipeline Ready |

> **Key Takeaway**: Preserving negation tokens and introducing $(1, 2)$ n-grams elevated the classical baseline by **+1.74%**, achieving **88.98%** and demonstrating the importance of domain-adapted feature engineering.

---

## Project Structure

```text
IMDB-Sentiment-Research/
│
├── data/
│   ├── raw/aclImdb/              # Official IMDb dataset (train & test splits)
│   └── processed/                # Compiled and preprocessed CSVs
│
├── deployment/
│   └── app.py                    # Streamlit web app (dual-model inference & SHAP)
│
├── models/
│   ├── baseline_pipeline.pkl     # Serialized end-to-end TF-IDF + Logistic Regression
│   ├── tfidf_vectorizer.pkl      # Standalone vectorizer
│   └── distilbert_imdb/          # Fine-tuned DistilBERT weights & tokenizer
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis & length distributions
│   ├── 02_Text_Cleaning.ipynb     # Contraction expansion & text sanitization
│   ├── 03_Feature_Engineering.ipynb # Negation-preserving TF-IDF experiments
│   ├── 04_Baseline_Model.ipynb   # Model comparisons (LR, LinearSVC, Naive Bayes)
│   ├── 05_Error_Analysis.ipynb   # False positive/negative deep dive
│   ├── 06_DistilBERT.ipynb       # DistilBERT fine-tuning workflow
│   └── 07_DeBERTa_FineTuning.ipynb # SOTA DeBERTa-v3 training with smart truncation
│
├── reports/
│   └── error_analysis_summary.md # Detailed linguistic failure mode report
│
├── src/
│   ├── train_baseline.py         # Standalone script to train & serialize baseline
│   └── evaluate.py               # Automated benchmark evaluation suite
│
├── requirements.txt              # Project dependencies
└── README.md                     # Research documentation
```

---

## Interactive Web Application

A live interactive web application is deployed via Streamlit Community Cloud:

* **Dual-Engine Inference**: Select between **Optimized TF-IDF Baseline** (instant, CPU-friendly), **DistilBERT** (deep learning), or **Side-by-Side Comparison**.
* **Quick-Test Presets**: One-click test reviews designed to stress-test models (Sarcastic Criticism, Mixed Nuance, Enthusiastic Praise, Decisive Negative).
* **Model Explainability (SHAP)**: Interactive token-level visualization showing which words drive positive (red) vs. negative (blue) predictions.
* **Token Length Indicator**: Real-time word and character counter tracking transformer context capacity.

To run locally:
```bash
streamlit run deployment/app.py
```

---

## Reproduction & Usage

### 1. Train the Optimized Baseline Pipeline
Trains the negation-aware TF-IDF vectorizer and tuned Logistic Regression model, outputting performance metrics and saving `models/baseline_pipeline.pkl`:
```bash
python src/train_baseline.py
```

### 2. Run the Benchmark Evaluation Suite
Evaluates both the serialized baseline pipeline and DistilBERT model on the holdout test partition:
```bash
python src/evaluate.py
```

### 3. Fine-Tune DeBERTa-v3 on GPU
Open `notebooks/07_DeBERTa_FineTuning.ipynb` in Google Colab or on a local GPU machine. The notebook features **Smart Head + Tail Truncation** (128 tokens head + 382 tokens tail), FP16 mixed precision, and early stopping.

---

## Linguistic Error Analysis Highlights

A systematic analysis of false positives and false negatives identified four dominant linguistic challenges:

1. **Sarcasm & Irony**: Superficial praise (*"stroke of genius"*, *"masterpiece"*) masking caustic criticism.
2. **Sentiment Shifts & Contrastive Conjunctions**: Reviews spending 80% praising the acting before delivering a negative verdict (*"however, the script was atrocious"*).
3. **Sequence Truncation**: Standard 256-token truncation inadvertently discards the reviewer's concluding sentence where the final verdict resides.
4. **Double Negatives**: Complex phrasing (*"not uninteresting"*, *"not without charm"*) resolved via negation preservation.

See [`reports/error_analysis_summary.md`](reports/error_analysis_summary.md) for the full report.

---

## Author

**Ranjit Das**  
*IMDb Sentiment Analysis Research Project*
