import pandas as pd             
import numpy as np               
import string                   
import matplotlib.pyplot as plt  
from textblob import TextBlob    
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier   
from sklearn.metrics import classification_report     

#This was to Load the fake and real news datasets.
#These datasets contain article text that we will analyze for linguistic patterns,
#and not content and this helps avoid topic bias and focuses on stylistic differences.
fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')

#We gave binary labels: 0 = fake, 1 = real.
#Because we needed labels early so they can be merged cleanly into a single dataframe.
fake_news_df['label'] = 0
real_news_df['label'] = 1

#Combined both datasets into one unified dataframe.
#This makes preprocessing easier because we operate on one object.
df = pd.concat([fake_news_df, real_news_df], ignore_index=True)

#Instead of trying to classify articles using topic and keywords which can end up biasing the model,
#we extracted structural and linguistic features. Since these tend to generalize better,
#especially for fake-news detection where writing style is more revealing than the subject matter.

#We calculated word count for each article since fake articles 
#sometimes exaggerate, use long paragraphs, or overly concise structures.
df['Word_Count'] = df['text'].apply(lambda x: len(str(x).split()))

#This is the Subjectivity Score for each article:
#TextBlob gives a score between 0 (objective) and 1 (subjective) for each article.
#Fake news often uses more emotional or opinionated language, so this feature may be predictive.
df['Subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Polarity Score which measures the emotional tone of the article:
# Polarity ranges from -1 (negative) to +1 (positive).
# Fake news often uses more emotionally extreme tone — this feature helps capture that.
df['Polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)


#The idea here is that fake news often uses more punctuation for drama:
#Excessive "!!!"
#Suspenseful "??"
#More commas in run-on sentences
#Counting punctuation helps capture this stylistic fingerprint.

def count_punctuation(text):
    #We count how many punctuation characters appear in the article.
    #And by using 'string.punctuation' gives us a clean, standard set of punctuation marks.
    count = sum([1 for char in str(text) if char in string.punctuation])
    return count

df['Punctuation_Count'] = df['text'].apply(count_punctuation)

#Punctuation Density:
#This normalizes punctuation usage by article length.
#Without normalization, long articles would naturally have more punctuation,
#so we compute punctuation per word a stylistic intensity score.
df['Punctuation_Density'] = np.divide(
    df['Punctuation_Count'],
    df['Word_Count']
).fillna(0).replace([np.inf, -np.inf], 0)


#Average word length tells us something about the sophistication
#or simplicity of the writing. Fake news sometimes uses simpler words
#to appeal to wider audiences or create emotional immediacy.
df['Avg_Word_Length'] = df['text'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) 
              if len(str(x).split()) > 0 else 0
)


#Target variable (0 or 1)
y = df['label']

#Select only the engineered stylistic features.
#We intentionally avoid raw text features to focus on writing patterns,
#which makes the model light, fast, and interpretable.
X = df[['Word_Count', 'Subjectivity', 'Polarity', 
        'Punctuation_Density', 'Avg_Word_Length']]

#Split the dataset:
#80% training
#20% testing
#Using a random_state guarantees reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Initialize a Random Forest model:
#We choose Random Forest because it:
#Handles non-linear relationships
#Provides feature importance scores
#Is robust to noise
#Performs well even on small feature sets
model = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the model on the engineered features
model.fit(X_train, y_train)

#Extract feature importance from the trained model.
#This is useful for understanding which stylistic cues matter most.
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

#THE VISUALIZATION

plt.figure(figsize=(10, 8))
feature_importances.plot(kind='bar', color='skyblue')

plt.title('Feature Importance in Fake News Detection')
plt.ylabel('Importance Score')
plt.xlabel('Stylistic/Structural Feature')
plt.xticks(rotation=0)
plt.grid(axis='y')

plt.savefig('feature_importance.png') # Save visual for reports or presentations
plt.show()