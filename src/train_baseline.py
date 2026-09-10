import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Negation words to keep in vocabulary (critical for sentiment analysis)
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nowhere",
    "hardly", "scarcely", "barely", "cannot", "without", "against"
}
# Filtered stopwords that do NOT strip negation
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - NEGATION_WORDS)

def get_data_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "processed", "train_reviews_clean.csv")
    model_output_path = os.path.join(project_root, "models", "baseline_pipeline.pkl")
    return data_path, model_output_path

def train_and_evaluate(data_path, model_output_path):
    print(f"Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Check nulls
    df["clean_review"] = df["clean_review"].fillna("")
    X = df["clean_review"]
    y = df["label"]

    print(f"Dataset size: {len(df)} samples")
    print(f"Class distribution:\n{y.value_counts()}")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    print("\nBuilding Optimized Baseline Pipeline:")
    print("  - TfidfVectorizer: ngram_range=(1, 2), sublinear_tf=True, min_df=3, negation-preserving stopwords")
    print("  - Classifier: LogisticRegression(C=2.5, max_iter=1000, solver='saga', random_state=42)")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.85,
            stop_words=CUSTOM_STOP_WORDS
        )),
        ("clf", LogisticRegression(
            C=2.5,
            max_iter=1000,
            solver="saga",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\nFitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print("=" * 50)
    print(f"Optimized Baseline Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save pipeline
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"\nSaved serialized pipeline to: {model_output_path}")
    
    # Save vectorizer separately for backward compatibility if needed
    vectorizer_path = os.path.join(os.path.dirname(model_output_path), "tfidf_vectorizer.pkl")
    joblib.dump(pipeline.named_steps["tfidf"], vectorizer_path)
    print(f"Updated tfidf_vectorizer at: {vectorizer_path}")

    return acc, pipeline

if __name__ == "__main__":
    data_path, model_path = get_data_paths()
    train_and_evaluate(data_path, model_path)
