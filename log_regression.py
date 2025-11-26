import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Loading and Labeling the Data
fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')

fake_news_df['label'] = 0 #This is Fake
real_news_df['label'] = 1 #This is Real

pattern_to_remove = r'^[A-Z/,\s]+\s*\(REUTERS\)\s*-\s*'
real_news_df['text_clean'] = real_news_df['text'].astype(str).str.replace(
    pattern_to_remove, 
    '', 
    regex=True, 
    flags=re.IGNORECASE
)
real_news_df.drop('text', axis=1, inplace=True)
real_news_df.rename(columns={'text_clean': 'text'}, inplace=True)

df = pd.concat([fake_news_df, real_news_df]).sample(frac=1).reset_index(drop=True)

#Selecting the Features and Target
X = df['text']
y = df['label']

# Training and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Vectorizing tfidf
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# The Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Predictions and Evaluation
y_pred = lr_model.predict(X_test_tfidf)
y_pred_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]  # Probability of class 1 (Real)

accuracy = accuracy_score(y_test, y_pred)

'''
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
'''

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.savefig("Confusion_Matrix.png")
plt.show()

# Distribution of Prediction Probabilities
plt.figure(figsize=(8, 5))
plt.hist(y_pred_proba, bins=30, edgecolor='black')
plt.title("Distribution of Prediction Probabilities")
plt.xlabel("Predicted Probability (Real News)")
plt.ylabel("Frequency")
plt.savefig("Distribution_of_Prediction_Probabilities.png")
plt.show()

# Prediction Probabilities by Actual Label
plt.figure(figsize=(8, 5))
plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='Fake News', color='red')
plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Real News', color='green')
plt.title("Prediction Probabilities by Actual Label")
plt.xlabel("Predicted Probability (Real News)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("Probabilities_by_Label.png")
plt.show()

# Prediction Distribution
plt.figure(figsize=(8, 5))
prediction_counts = pd.Series(y_pred).value_counts().sort_index()
plt.bar(['Fake (0)', 'Real (1)'], prediction_counts.values, color=['red', 'green'], alpha=0.7)
plt.title("Distribution of Predictions")
plt.xlabel("Predicted Label")
plt.ylabel("Count")
plt.savefig("Prediction_Distribution.png")
plt.show()

#Logistic Regression done by Kavin Ramesh, Issue with pushing to GitHub