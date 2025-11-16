import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords #stopwords for english
from nltk.tokenize import word_tokenize #word tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words_english = stopwords.words('english')

fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')

#Columns in each dataset ['title', 'text', 'subject', 'date']

#Step 1: Cleaning the Fake News Dataset

fn_text_columns = ['title', 'text', 'subject']

for col in fn_text_columns:
    fake_news_df[col] = (
        fake_news_df[col]
        .astype(str)
        .str.lower()
        .str.replace("[^a-zA-Z ]", " ", regex=True)  # removes apostrophes too
    )

for col in fn_text_columns:
    fake_news_df[col] = (
        fake_news_df[col]
        .apply(word_tokenize)
        .apply(lambda x: [word for word in x if word not in stop_words_english])
        .apply(lambda x: ' '.join(x))
    )

fake_news_df = fake_news_df[fake_news_df['text'].str.len() > 0]

# Stemming and Lemmatization was intentionally not used as our datasets did not come close to being large enough to require it.
#print(fake_news_df.head())

#Stemming and Lammatization was not relevant as our datasets did not come close to being large enough to require it.

#Step 1a: Cleaning in the real news Dataset

rn_text_columns = ['title', 'text', 'subject']

for col in rn_text_columns:
    real_news_df[col] = (
        real_news_df[col]
        .astype(str)
        .str.lower()
        .str.replace("[^a-zA-Z ]", " ", regex=True)  # removes apostrophes too
    )

for col in rn_text_columns:
    real_news_df[col] = (
        real_news_df[col]
        .apply(word_tokenize)
        .apply(lambda x: [word for word in x if word not in stop_words_english])
        .apply(lambda x: ' '.join(x))
    )

real_news_df = real_news_df[real_news_df['text'].str.len() > 0]

#print(real_news_df.head())
#Step 2: Finding Patterns in Words of the Fake News Dataset