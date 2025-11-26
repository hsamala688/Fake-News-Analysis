#Our NLP model came back with very impressive results, too good results. So we are going through and trying to verify 
# the datasets and the quality that we trained it on
import pandas as pd
import sklearn as sk
import re

fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')
fake_news_df['label'] = 0
real_news_df['label'] = 1

#First analysis: 
df = pd.concat([fake_news_df, real_news_df], ignore_index=True)
stratify = df['label']

initial_row_count = len(df)
df_dedup = df.drop_duplicates(subset = ['title', 'text'], keep ='first')
dedup_row_count = len(df_dedup)

print(f"Original Row Count: {initial_row_count}")
print(f"Row Count After Deduplication: {dedup_row_count}")

if initial_row_count != dedup_row_count:
    print(f"Found and removed {initial_row_count - dedup_row_count} exact duplicates.")

'''Basic results are checking for duplicates in the combined dataset:
Original Row Count: 44898
Row Count After Deduplication: 39105
Found and removed 5793 exact duplicates.
This is a significant number of duplicates, indicating that the datasets had overlapping entries. This could easily have
affected training

Next step was to check if the date columns had an impact on the training data'''

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)
temporal_bias = df.groupby(df['date'].dt.to_period('M'))['label'].agg(['count', 'mean']).reset_index()
temporal_bias.columns = ['date', 'article_count', 'fake_news_ratio']
temporal_bias['date_month'] = temporal_bias['date'].astype(str)
print(temporal_bias)

'''After performing temporal analysis, we found that there was cetainly some concerning results as between 2015-05 and 2017-12
there was a fake_news_ration of 0.0 for EVERY MONTH. This indicates a high potential temporal bias in the dataset

Here is also something that is very concerning, that we caught. Apparently in every single text column in the true dataset, 
there is a 'Reuters' tag at the start of every article. This is a source leakage'''

pattern = r'\(?reuters\)?'
tagged_articles = real_news_df[real_news_df['text'].astype(str).str.contains
                               (pattern, regex=True, flags=re.IGNORECASE)].copy()
print(f"Reuters Tags: Found {len(tagged_articles)} articles)")
print(f"Visualizing the Location of the Reuters Tag (Found {len(tagged_articles)} articles)")
print("First 5 articles for visual inspection:")

if not tagged_articles.empty:
    for i, row in tagged_articles.head(5).iterrows():
        print(f"\n==================== Article ID: {i} ====================")
        # Use a regex findall to highlight where the tag is found (if possible)
        text_with_highlight = re.sub(pattern, lambda m: f'**<{m.group(0).upper()}>**', row['text'], flags=re.IGNORECASE)
        print(text_with_highlight)
else:
    print("\nError: The 'true.csv' file does not contain the word 'Reuters' anywhere. Check the file content.")

'''So after preforming analysis on the true.csv we found that almost all the articles in the dataset had a 'Reuters' 
tag at the start of each article. This is a significant source leakage and likely contributed to the 
high accuracy of the model.
So we have to remove this tag from all articles in the true dataset before retraining the model.'''

real_news_df = pd.read_csv('True.csv')
pattern_to_remove = r'^[A-Z/,\s]+\s*\(REUTERS\)\s*-\s*'
real_news_df['text_clean'] = real_news_df['text'].astype(str).str.replace(
    pattern_to_remove, 
    '', 
    regex=True, 
    flags=re.IGNORECASE
)
print("--- Verification of Cleaned Text ---")
print(real_news_df['text_clean'].head().str[:100])
print(real_news_df['text'].head().str[:100])

'''After adding these cleaning steps to the true dataset, we can no retrain the model and see if the accuracy changes at all.
Proof of Cleaning:
--- Verification of Cleaned Text ---
Old Text Samples:
0    WASHINGTON (Reuters) - The head of a conservat...
1    WASHINGTON (Reuters) - Transgender people will...
2    WASHINGTON (Reuters) - The special counsel inv...
3    WASHINGTON (Reuters) - Trump campaign adviser ...
4    SEATTLE/WASHINGTON (Reuters) - President Donal...

Cleaned Text Samples:
0    The head of a conservative Republican faction ...
1    Transgender people will be allowed for the fir...
2    The special counsel investigation of links bet...
3    Trump campaign adviser George Papadopoulos tol...
4    President Donald Trump called on the U.S. Post...
'''

print(fake_news_df['subject'].value_counts())
print(real_news_df['subject'].value_counts())


'''After all this another issue was identified in the cleaning steps of the datasets.
The problem now was with the subjects column where there classification labels between the two datasets were not aligned
True.csv subjects included:
politicsnews    11272
worldnews       10145

Fake.csv subjects included:
subject
news               9050
politics           6432
left news          4309
government news    1498
us news             783
middle east         778

As a result of this finding we need to just drop the subject column from the training datasets and then retrain the model
'''

#Comments and Code done by Hayden