import os
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_validation_data():
    """Loads the test partition from the processed dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "processed", "train_reviews_clean.csv")
    
    df = pd.read_csv(data_path)
    df["clean_review"] = df["clean_review"].fillna("")
    X = df["clean_review"]
    y = df["label"]
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

def evaluate_baseline(X_test, y_test):
    """Evaluates the serialized baseline pipeline."""
    print("\n" + "=" * 50)
    print("1. Evaluating Optimized TF-IDF Baseline Pipeline...")
    print("=" * 50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pipeline_path = os.path.join(project_root, "models", "baseline_pipeline.pkl")
    
    if not os.path.exists(pipeline_path):
        print(f"Serialized pipeline not found at {pipeline_path}. Please run src/train_baseline.py first.")
        return None

    try:
        pipeline = joblib.load(pipeline_path)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%")
        print(f"F1-Score:  {f1 * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
        
        return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    except Exception as e:
        print(f"Error evaluating baseline: {e}")
        return None

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_distilbert(X_test, y_test):
    """Evaluates the fine-tuned DistilBERT transformer model."""
    print("\n" + "=" * 50)
    print("2. Evaluating DistilBERT Transformer Model...")
    print("=" * 50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Try both nested and standard path
    model_paths = [
        os.path.join(project_root, "models", "distilbert_imdb", "distilbert_imdb"),
        os.path.join(project_root, "models", "distilbert_imdb")
    ]
    
    model_path = None
    for p in model_paths:
        if os.path.exists(os.path.join(p, "model.safetensors")) or os.path.exists(os.path.join(p, "pytorch_model.bin")):
            model_path = p
            break
            
    if model_path is None:
        print("DistilBERT model weights not found in models/distilbert_imdb/. Skipping transformer evaluation.")
        return None

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}.")
        
        print("Tokenizing test samples (max_length=256)...")
        test_encodings = tokenizer(
            X_test.tolist(), truncation=True, padding=True, max_length=256
        )
        
        test_dataset = IMDbDataset(test_encodings, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        print(f"Running batch inference on {len(X_test)} reviews...")
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
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary")
        
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%")
        print(f"F1-Score:  {f1 * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, predictions, target_names=["Negative", "Positive"]))
        
        return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    except Exception as e:
        print(f"Error evaluating DistilBERT: {e}")
        return None

if __name__ == "__main__":
    print("Loading test partition (5,000 samples)...")
    X_test, y_test = load_validation_data()
    print(f"Test samples ready: {len(X_test):,}")

    base_metrics = evaluate_baseline(X_test, y_test)
    dist_metrics = evaluate_distilbert(X_test, y_test)

    print("\n" + "=" * 55)
    print("SUMMARY OF BENCHMARK RESULTS")
    print("=" * 55)
    rows = []
    if base_metrics:
        rows.append({
            "Model": "Optimized TF-IDF Baseline",
            "Accuracy": f"{base_metrics['Accuracy']*100:.2f}%",
            "F1-Score": f"{base_metrics['F1']*100:.2f}%",
            "Precision": f"{base_metrics['Precision']*100:.2f}%",
            "Recall": f"{base_metrics['Recall']*100:.2f}%"
        })
    if dist_metrics:
        rows.append({
            "Model": "Fine-Tuned DistilBERT",
            "Accuracy": f"{dist_metrics['Accuracy']*100:.2f}%",
            "F1-Score": f"{dist_metrics['F1']*100:.2f}%",
            "Precision": f"{dist_metrics['Precision']*100:.2f}%",
            "Recall": f"{dist_metrics['Recall']*100:.2f}%"
        })
    if rows:
        summary_df = pd.DataFrame(rows)
        print(summary_df.to_string(index=False))
    print("=" * 55)
