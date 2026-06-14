import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_data():
    df = pd.read_csv("data/processed/train_reviews_clean.csv")
    X = df["clean_review"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

from sklearn.linear_model import LogisticRegression

def evaluate_baseline(X_train, X_test, y_train, y_test):
    print("Evaluating TF-IDF + Logistic Regression Baseline...")
    try:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train_vec, y_train)
        
        preds = lr_model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"Baseline Accuracy: {acc * 100:.2f}%")
        return acc
    except Exception as e:
        print(f"Failed to evaluate baseline: {e}")
        return 0.0

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_distilbert(X_test, y_test):
    print("Evaluating DistilBERT...")
    model_path = "models/distilbert_imdb/distilbert_imdb"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Please unzip it first.")
        return 0.0

    try:
        # Load tokenizer and model
        # Try loading from local path, fallback to base uncased for tokenizer if local is missing tokenizer files
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print(f"Model loaded onto {device}. Tokenizing test data...")
        test_encodings = tokenizer(
            X_test.tolist(), truncation=True, padding=True, max_length=256
        )
        
        test_dataset = IMDbDataset(test_encodings, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        print("Running inference...")
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                
        acc = accuracy_score(y_test, predictions)
        print(f"DistilBERT Accuracy: {acc * 100:.2f}%")
        return acc

    except Exception as e:
        print(f"Failed to evaluate DistilBERT: {e}")
        return 0.0

if __name__ == "__main__":
    print("Loading test data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Test data size: {len(X_test)}")
    
    base_acc = evaluate_baseline(X_train, X_test, y_train, y_test)
    dist_acc = evaluate_distilbert(X_test, y_test)
    
    print("\n" + "="*30)
    print("RESULTS:")
    print(f"TF-IDF + LR Baseline: {base_acc * 100:.2f}%")
    print(f"DistilBERT Model: {dist_acc * 100:.2f}%")
    max_acc = max(base_acc, dist_acc)
    print(f"Maximum Accuracy Achieved: {max_acc * 100:.2f}%")
    print("="*30)
