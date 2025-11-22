import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import string
from textblob import TextBlob

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
correlation_df = pd.concat([X, y], axis=1)
correlation_matrix = correlation_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm',fmt=".2f",linewidths=.5,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap of Features in Fake and Real News Dataset')
plt.savefig('correlation_heatmap.png')
plt.show()
##This is our heatmap correlation 