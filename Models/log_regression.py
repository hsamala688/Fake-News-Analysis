import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Loading and Labeling the Data
fake_news_df = pd.read_csv("Fake.csv")
real_news_df = pd.read_csv("True.csv")

fake_news_df["label"] = 0  # This is Fake
real_news_df["label"] = 1  # This is Real

"""
We defined a regex pattern that removes the common newswire header found in many real news articles.
Since Many Real News dataset entries begin with something like:
"WASHINGTON (REUTERS) - ..." or "NEW YORK, (Reuters) - ..."
These prefixes are not part of the article content and would bias text processing
because they add location names and the word “REUTERS” at the start of every article.
The pattern does the following:
^[A-Z/,\s]+      → matches city names or region identifiers written in all caps
  (e.g., 'WASHINGTON', 'NEW YORK,', 'LONDON/BEIJING')
 \s*\(REUTERS\)\s*→ matches the literal "(REUTERS)" tag, with optional spaces
  -\s*             → matches the dash after the tag, which usually separates the header from the main text
"""

pattern_to_remove = r"^[A-Z/,\s]+\s*\(REUTERS\)\s*-\s*"

"""
We applied the regex pattern to remove the Reuters-style header from each article.
We converted the text to string first (astype(str)) to avoid errors from any missing or non-string values.
regex=True tells pandas to interpret the first argument as a regex.
flags=re.IGNORECASE ensures "(REUTERS)" matches regardless of capitalization variants.
"""
real_news_df["text_clean"] = (
    real_news_df["text"]
    .astype(str)
    .str.replace(pattern_to_remove, "", regex=True, flags=re.IGNORECASE)
)
# We remove the old 'text' column now that a cleaned version exists.
# This keeps the dataset tidy and prevents confusion between raw and cleaned versions.
real_news_df.drop("text", axis=1, inplace=True)

# Renamed 'text_clean' back to 'text' so the rest of the pipeline can continue unchanged.
# Our model, preprocessing, vectorizers, and future steps expect a column literally named 'text'.
real_news_df.rename(columns={"text_clean": "text"}, inplace=True)


# We’re combining both the fake and real news DataFrames into a single dataset.
# Using pd.concat([...]) stacks them vertically so we end up with one unified table
# that contains:
# - all fake articles (label = 0)
# - all real articles (label = 1)
# We need everything merged before creating our train/test split so the model
# learns from both classes together.

df = pd.concat([fake_news_df, real_news_df]).sample(frac=1).reset_index(drop=True)

# We shuffle the entire dataset using .sample(frac=1).
# Without shuffling, all fake articles would appear first and all real articles
# would appear last, which could create bias or produce an uneven train/test split.
# Shuffling ensures the examples are randomly mixed.

# After shuffling, we reset the index so the DataFrame stays clean and sequential.
# reset_index(drop=True) gives us a fresh 0-to-n index and prevents the old index
# from being added back as a column.

# Selecting the Features and Target
X = df["text"]
y = df["label"]

# Training and Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

"""
Vectorizing using TF-IDF
Before training, we need to convert raw text into numerical features.
We use TfidfVectorizer, which transforms text into weighted word scores based on how important each word is within the document and across the dataset.
stop_words='english' removes very common words ("the", "and", etc.) that don't help the model distinguish fake vs. real articles.
max_features=5000 limits the vocabulary to the top 5000 most informative words, which helps reduce noise and improves training speed.
"""
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# The Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Predictions and Evaluation
# After training, we use the model to make predictions on the test set.
# y_pred contains the actual predicted labels (0 = fake, 1 = real) for each test article.
y_pred = lr_model.predict(X_test_tfidf)

# We also calculate predicted probabilities for class 1 (Real news).
# [:, 1] extracts the probability that each article belongs to class 1.
# This is useful later for ROC curves, threshold tuning, or probability-based evaluations.
y_pred_proba = lr_model.predict_proba(X_test_tfidf)[
    :, 1
]  # Probability of class 1 (Real)

accuracy = accuracy_score(y_test, y_pred)

"""
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
"""

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.savefig("Confusion_Matrix.png")
plt.show()

# Distribution of Prediction Probabilities
plt.figure(figsize=(8, 5))
plt.hist(y_pred_proba, bins=30, edgecolor="black")
plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Predicted Probability (Real News)")
plt.ylabel("Frequency")
plt.savefig("Distribution_of_Prediction_Probabilities.png")
plt.show()

# Prediction Probabilities by Actual Label
plt.figure(figsize=(8, 5))
plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label="Fake News", color="red")
plt.hist(
    y_pred_proba[y_test == 1], bins=30, alpha=0.5, label="Real News", color="green"
)
plt.title("Prediction Probabilities by Actual Label")
plt.xlabel("Predicted Probability (Real News)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("Probabilities_by_Label.png")
plt.show()

# Prediction Distribution
plt.figure(figsize=(8, 5))
prediction_counts = pd.Series(y_pred).value_counts().sort_index()
plt.bar(
    ["Fake (0)", "Real (1)"],
    prediction_counts.values,
    color=["red", "green"],
    alpha=0.7,
)
plt.title("Distribution of Predictions")
plt.xlabel("Predicted Label")
plt.ylabel("Count")
plt.savefig("Prediction_Distribution.png")
plt.show()

# Logistic Regression done by Kavin Ramesh, Issue with pushing to GitHub
