import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report

fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')
fake_news_df['label'] = 0
real_news_df['label'] = 1

df = pd.concat([fake_news_df, real_news_df], ignore_index=True)
df['Word_Count'] = df['text'].apply(lambda x: len(str(x).split()))
df['Subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
df['Polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
def count_punctuation(text):
    count = sum([1 for char in str(text) if char in string.punctuation])
    return count

df['Punctuation_Count'] = df['text'].apply(count_punctuation)
df['Punctuation_Density'] = np.divide(df['Punctuation_Count'], df['Word_Count']).fillna(0).replace([np.inf, -np.inf], 0)
df['Avg_Word_Length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)

y = df['label']
X = df[['Word_Count', 'Subjectivity', 'Polarity', 'Punctuation_Density', 'Avg_Word_Length']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color = 'skyblue')
plt.title('Feature Importance in Fake News Detection')
plt.ylabel('Importance Score')
plt.xlabel('Stylistic/Structural Feature')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('feature_importance.png')
plt.show()