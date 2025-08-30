import argparse
import json
import os
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from joblib import dump

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # remove urls/emails
    s = re.sub(r"(https?://\S+)|(\w+@\w+\.\w+)", " ", s)
    # remove numbers
    s = re.sub(r"\d+", " ", s)
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_dataset(path: str):
    df = pd.read_csv(path, encoding="latin-1")
    # Try common schemas
    if {"label", "text"}.issubset(df.columns):
        df = df[["label", "text"]].dropna()
        df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0, "non-spam": 0, "nonspam": 0})
    elif {"v1", "v2"}.issubset(df.columns):
        # SMS Spam Collection style
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})
    else:
        # Try to guess: first column is label, second is text
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]
        df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})
    df = df.dropna()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df[df["label"].isin([0, 1])]
    return df

def main(args):
    df = load_dataset(args.data)
    X = df["text"].tolist()
    y = df["label"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200, n_jobs=1))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    dump(pipe, args.model)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples": {
            "train": len(X_train),
            "test": len(X_test)
        },
        "vectorizer": {
            "max_features": args.max_features,
            "ngram_range": [1, 2]
        },
        "model": "LogisticRegression(max_iter=200)"
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to:", args.model)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--model", default=os.path.join("models", "model.joblib"))
    parser.add_argument("--report", default=os.path.join("models", "metrics.json"))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_features", type=int, default=20000)
    args = parser.parse_args()
    main(args)
