# IMDb Sentiment Analysis Research

## Overview

This project explores sentiment analysis on the IMDb movie review dataset using both traditional NLP techniques and transformer-based deep learning models.

The objective is to classify movie reviews as positive or negative and compare the performance of a TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT model.

---

## Dataset

* Dataset: IMDb Movie Reviews
* Total Reviews: 25,000
* Classes:

  * Positive
  * Negative

---

## Project Workflow

1. Exploratory Data Analysis (EDA)
2. Text Cleaning and Preprocessing
3. Feature Engineering using TF-IDF
4. Logistic Regression Baseline Model
5. Error Analysis
6. DistilBERT Fine-Tuning
7. Model Comparison

---

## Project Structure

```text
IMDB-Sentiment-Research/
│
├── data/
├── deployment/
├── models/
├── notebooks/
├── reports/
├── src/
└── README.md
```

---

## Models Evaluated

### TF-IDF + Logistic Regression

Accuracy: **87.24%**

### DistilBERT

Accuracy: **89.74%**

---

## Error Analysis Findings

Common error categories identified:

* Mixed Sentiment
* Sentiment Shift
* Context Misunderstanding
* Negation Handling
* Humor and Sarcasm

These findings motivated the transition from TF-IDF features to transformer-based language models.

---

## Results

| Model                        | Accuracy |
| ---------------------------- | -------- |
| TF-IDF + Logistic Regression | 87.24%   |
| DistilBERT                   | 89.74%   |

DistilBERT improved performance by approximately **2.5 percentage points** over the baseline model.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Hugging Face Transformers
* PyTorch
* Google Colab
* Jupyter Notebook

---

## Future Improvements

* Streamlit deployment
* Hyperparameter tuning
* RoBERTa and DeBERTa experiments
* Model monitoring and explainability

---

## Author

Ranjit Das
