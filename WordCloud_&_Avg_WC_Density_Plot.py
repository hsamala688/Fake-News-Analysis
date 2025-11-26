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

#We Loaded Fake and Real news datasets.
#We label them separately and combine them later.
fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')

#We Assign binary labels (0 = Fake, 1 = Real) so the model
#can later distinguish the two groups during analysis.
fake_news_df['label'] = 0
real_news_df['label'] = 1


#WordClouds originally had several issues:
# -Floating “S” characters caused by "'s" splitting incorrectly.
# -Apostrophes, commas, and punctuation turning into their own tokens.
# -WordCloud mishandling case-sensitive words (e.g., "US", "BREAKING").
#
#This custom preprocessing fixes these issues by:
# -Keeping case sensitivity (important because fake news often uses ALL CAPS)
# -Removing stopwords so only meaningful content appears in the cloud
# -Removing punctuation tokens that generate noise
# -Removing possessive endings to avoid stray "S" artifacts
def preprocess_text_case_sensitive_v2(text, custom_stopwords):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = []
    
    for word in tokens:
        #We remove stopwords so WordCloud emphasizes real content words
        if word.lower() in custom_stopwords:
            continue

        #We filter out punctuation tokens that break WordCloud visualizations
        if word in ("'", "’", ".", ","):
            continue

        #We remove possessive endings ('s, ’s) which caused “S” to appear incorrectly
        if word in ("'s", "’s", "'S", "’S"):
            continue
        
        filtered_tokens.append(word)
    
    return ' '.join(filtered_tokens)


#We combine all article text into one giant string per class,
#clean it, and prepare it for WordCloud visualization.
fake_text = preprocess_text_case_sensitive_v2(
    ' '.join(fake_news_df['text'].astype(str).tolist()),
    set(stopwords.words('english'))
)

real_text = preprocess_text_case_sensitive_v2(
    ' '.join(real_news_df['text'].astype(str).tolist()),
    set(stopwords.words('english'))
)

#WordCloud regex handles apostrophes and complex tokens correctly.
#Default WordCloud splits “don’t” → “don” + “t”, so we override with a better regex.
fake_wordcloud = WordCloud(
    width=1200, height=600, background_color='white',
    regexp=r"\w[\w']*\w|\w"
).generate(fake_text)

real_wordcloud = WordCloud(
    width=1200, height=600, background_color='white',
    regexp=r"\w[\w']*\w|\w"
).generate(real_text)

# WORD CLOUD VISUALIZATION
plt.figure(figsize=(16, 8))

#Fake News WordCloud
plt.subplot(1, 2, 1)
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Fake News Articles')
plt.axis('off')

#Real News WordCloud
plt.subplot(1, 2, 2)
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Real News Articles')
plt.axis('off')

#Save and display WordCloud figure
plt.savefig('wordclouds_fake_vs_real.png')
plt.show()

# WORD COUNT ANALYSIS FOR DISTRIBUTION COMPARISON
# We compute word counts because fake news often follows different stylistic patterns:
# -Fake articles are often shorter, punchier, or less detailed.
# -Real journalism tends to be longer, more structured, and more in-depth.

fake_news_df['word_count'] = fake_news_df['text'].astype(str).apply(lambda x: len(x.split()))
real_news_df['word_count'] = real_news_df['text'].astype(str).apply(lambda x: len(x.split()))

#KDE plots let us visualize smooth probability distributions and compare styles.
plt.figure(figsize=(12, 6))

sns.kdeplot(fake_news_df['word_count'], label='Fake News', color='red', fill=True, alpha=0.5)
sns.kdeplot(real_news_df['word_count'], label='Real News', color='blue', fill=True, alpha=0.5)

plt.title('Word Count Distribution in Fake vs Real News Articles')
plt.xlabel('Word Count')
plt.ylabel('Density')
plt.legend()

#Cap x-axis to avoid extremely long articles flattening the plot and hiding differences
plt.xlim(0, 2000)
plt.grid()

#Save KDE distribution plot
plt.savefig('word_count_distribution.png')
plt.show()