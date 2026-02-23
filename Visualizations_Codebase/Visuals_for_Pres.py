import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ======================
# Loading and Labeling Data
# ======================
fake_news_df = pd.read_csv("Fake_copy.csv")
real_news_df = pd.read_csv("True_copy.csv")

fake_news_df["label"] = 0  # Fake
real_news_df["label"] = 1  # Real

# Clean Reuters headers
pattern_to_remove = r"^[A-Z/,\s]+\s*\(REUTERS\)\s*-\s*"
real_news_df["text_clean"] = (
    real_news_df["text"]
    .astype(str)
    .str.replace(pattern_to_remove, "", regex=True, flags=re.IGNORECASE)
)

real_news_df.drop("text", axis=1, inplace=True)
real_news_df.rename(columns={"text_clean": "text"}, inplace=True)

# Combine & shuffle dataset
df = pd.concat([fake_news_df, real_news_df]).sample(frac=1).reset_index(drop=True)

# ======================
# Train/Test Split
# ======================
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ======================
# TF-IDF Vectorization
# ======================
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ======================
# Logistic Regression Model + Runtime Measurement
# ======================

# ---- Training runtime ----
train_start = time.time()
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
train_end = time.time()

lr_train_runtime = train_end - train_start

# ---- Evaluation runtime ----
eval_start = time.time()
y_pred = lr_model.predict(X_test_tfidf)
y_pred_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]
eval_end = time.time()

lr_eval_runtime = eval_end - eval_start

# ======================
# Evaluation Metrics
# ======================
print("\n===== LOGISTIC REGRESSION RESULTS =====")
print(f"Training Runtime: {lr_train_runtime:.4f} seconds")
print(f"Evaluation Runtime: {lr_eval_runtime:.4f} seconds\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
