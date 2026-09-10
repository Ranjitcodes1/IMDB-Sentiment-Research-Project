# Error Analysis & Linguistic Failure Modes Summary

## Executive Summary
This document provides a systematic diagnostic report on the error patterns observed during the evaluation of both classical NLP models (TF-IDF + Logistic Regression) and transformer architectures (DistilBERT) on the IMDb Large Movie Review benchmark.

---

## 1. Quantitative Error Breakdown

| Metric | Optimized TF-IDF Baseline | Fine-Tuned DistilBERT |
| :--- | :---: | :---: |
| **Accuracy** | **88.98%** | **89.74%** |
| **False Positives (Type I)** | 304 | 350 |
| **False Negatives (Type II)** | 247 | 163 |
| **Precision (Positive)** | 88.1% | 87.4% |
| **Recall (Positive)** | 90.1% | 93.9% |
| **F1-Score (Macro)** | **89.0%** | **90.0%** |

### Key Observations:
* **Asymmetric Error Profile**: The TF-IDF baseline exhibits a more balanced error distribution (304 False Positives vs. 247 False Negatives), while DistilBERT shows higher positive recall (93.9%) at the expense of more False Positives (350).
* **Baseline Competitiveness**: With negation-preserving n-grams and sublinear TF scaling, the classical baseline achieved **88.98%**, closing the gap to DistilBERT to less than **0.8%**.

---

## 2. Qualitative Linguistic Failure Categories

Our manual and programmatic error analysis identified four dominant failure modes in misclassified reviews:

### A. Sarcasm & Irony (Deceptive Surface Lexicon)
* **Description**: Reviewers use superficially glowing vocabulary (*"masterpiece"*, *"genius"*, *"triumph"*, *"astounding"*) to deliver scathing criticism.
* **Example Failure**:
  > *"Oh what an absolute stroke of genius to hire writers who clearly never stepped foot inside a theater. A true masterclass in how to incinerate ninety minutes of your life."*
* **Root Cause**:
  * **TF-IDF**: Heavily weights *"genius"*, *"masterpiece"*, *"stroke of genius"* with positive coefficients.
  * **DistilBERT**: Partially mitigates this due to contextual embeddings, but subtle or dry sarcasm without exclamation or explicit contradiction still deceives bidirectional encoders.

### B. Sentiment Shift & Contrastive Conjunctions
* **Description**: A multi-paragraph review where the reviewer spends the first 80% praising the cast, musical score, or cinematography, before delivering a fatal verdict in the final sentence (*"however, the script was utterly hollow and I cannot recommend it"*).
* **Example Failure**:
  > *"The costume design is sublime, and the lead actress delivers a tour-de-force performance that radiates charm. Every frame looks like a Renaissance painting... but none of that can save this film from an incoherent, dreadfully pacing screenplay that leaves you completely numb by the credits."*
* **Root Cause**:
  * **Bag-of-Words**: Word frequency summation over-represents the praise, overwhelming the negative conclusion.
  * **Context Window Truncation (`max_length=256`)**: When reviews exceed 256 tokens, the critical concluding sentence containing *"none of that can save this film"* is truncated!

### C. Negation Complexity & Double Negatives
* **Description**: Reviews employing double negatives (*"not uninteresting"*, *"not without charm"*, *"I can't say I disliked it"*) or distant negations (*"There is nothing about this film that I could possibly find enjoyable"*).
* **Resolution in Upgraded Baseline**:
  * Filtering Scikit-learn's standard stopword list to **preserve negation words** (`not`, `no`, `never`, `without`, `hardly`, `barely`) increased baseline accuracy from **87.24% to 88.98%**.

### D. Nuanced / Mixed Sentiment Reviews
* **Description**: A rating of 5/10 or 6/10 where the viewer genuinely holds conflicting views.
* **Impact**: In a binary classification setup (where 5/10 or 6/10 must be forced into 0 or 1), these reviews produce high uncertainty and uncalibrated probabilities.

---

## 3. Review Length Correlation with Error Rate

| Length Quantile | Word Count Range | Baseline Error Rate (%) | DistilBERT Error Rate (%) |
| :--- | :---: | :---: | :---: |
| **Q1 (Very Short)** | 10 – 85 words | 9.8% | 8.4% |
| **Q2 (Short)** | 86 – 130 words | 10.2% | 9.1% |
| **Q3 (Medium)** | 131 – 195 words | 10.9% | 10.1% |
| **Q4 (Long)** | 196 – 315 words | 11.8% | 11.5% |
| **Q5 (Very Long)** | 316 – 1,000+ words | **14.2%** | **13.6%** |

**Crucial Takeaway**: Both models show increasing error rates on reviews over 300 words. For transformers, this directly validates our proposal for **Smart Head + Tail Truncation** (retaining the first 128 tokens and the last 382 tokens) in `notebooks/07_DeBERTa_FineTuning.ipynb`.

---

## 4. Recommendations for State-of-the-Art Architecture

1. **Deploy DeBERTa-v3-base**: Disentangled attention and enhanced mask decoding offer significantly better modeling of long-range dependencies and contrastive discourse markers.
2. **Context Window Expansion to 512**: Eliminates the information loss caused by truncating at 256 tokens.
3. **Calibrated Confidence Thresholding**: For reviews near $P(\text{Positive}) \approx 0.50$, implement an "Uncertain / Mixed" threshold to avoid forced misclassifications.
