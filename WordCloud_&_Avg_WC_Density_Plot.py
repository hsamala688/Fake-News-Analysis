import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import wordcloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud

fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')
fake_news_df['label'] = 0
real_news_df['label'] = 1

'''print(real_news_df['text'].value_counts())
print(fake_news_df['text'].value_counts())
print(fake_news_df['text'].describe())
print(real_news_df['text'].describe())'''

#First Visualization: Two Word Clouds for Fake vs Real News Articles
#Basic Cleanning for Word Cloud Visualization
#Basically we had a bunch of issues with floating S in the WordClouds so we had to make a custom text preprocessor
def preprocess_text_case_sensitive_v2(text, custom_stopwords):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = []
    
    for word in tokens:
        if word.lower() in custom_stopwords:
            continue
        
        if word in ("'", "’", ".", ","):
            continue
            
        if word in ("'s", "’s", "'S", "’S"):
            continue
        
        filtered_tokens.append(word)
    
    return ' '.join(filtered_tokens)


fake_text = preprocess_text_case_sensitive_v2(' '.join(fake_news_df['text'].astype(str).tolist()), set(stopwords.words('english')))
real_text = preprocess_text_case_sensitive_v2(' '.join(real_news_df['text'].astype(str).tolist()), set(stopwords.words('english')))
fake_wordcloud = WordCloud(width=1200, height=600, background_color='white',regexp=r"\w[\w']*\w|\w").generate(fake_text)
real_wordcloud = WordCloud(width=1200, height=600, background_color='white',regexp=r"\w[\w']*\w|\w").generate(real_text)
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Fake News Articles')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Real News Articles')
plt.axis('off')
plt.savefig('wordclouds_fake_vs_real.png')
plt.show()

#Second Visualization: Visualizing average Word Count Density in Real vs Fake datasets
fake_news_df['word_count'] = fake_news_df['text'].astype(str).apply(lambda x: len(x.split()))
real_news_df['word_count'] = real_news_df['text'].astype(str).apply(lambda x: len(x.split())) 
plt.figure(figsize=(12, 6))
sns.kdeplot(fake_news_df['word_count'], label='Fake News', color='red', fill=True, alpha=0.5)
sns.kdeplot(real_news_df['word_count'], label='Real News', color='blue', fill=True, alpha=0.5)
plt.title('Word Count Distribution in Fake vs Real News Articles')
plt.xlabel('Word Count')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, 2000)  # Limit x-axis for better visualization
plt.grid()
plt.savefig('word_count_distribution.png')
plt.show()