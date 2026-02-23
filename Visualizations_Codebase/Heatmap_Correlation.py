import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob

# The packages we needed to create this heatmap correlation

fake_news_df = pd.read_csv("Fake_copy.csv")
real_news_df = pd.read_csv("True_copy.csv")
fake_news_df["label"] = (
    0  # Again adding a column assigning all rows in this dataset the value of 0 indicating it is fake news
)
real_news_df["label"] = (
    1  # Again adding a column assigning all rows in this dataset the value of 1 indicating that it is real news
)
# Pandas used immediately

df = pd.concat([fake_news_df, real_news_df], ignore_index=True)
# Combining of the two datasets together to prep for the making of the correlation map

df["Word_Count"] = df["text"].apply(lambda x: len(str(x).split()))
"""
This creates a new column based from the original text column and then using the .apply we use lambda x to make
sure that it is applied to every row in that column and then we count how many words are in that text string and using the
split function add white space between every word
"""

df["Subjectivity"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
"""
Again we are creating a new column based from the text column. Using lambda to iterate through the entire column and
using TextBlob and the inherent property of sentiment from TextBlob to and then we extract that score from it
"""

df["Polarity"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
"""Same process as Subjectivity just using the TextBlob property of Polarity"""


def count_punctuation(text):
    count = sum([1 for char in str(text) if char in string.punctuation])
    return count


"""
First we define Text as the input then we make sure to convert everything to str and then for every character that
we detect in the text string (we can detect if the character is a punctuation mark using the string library punctuation).
We then create a list for each punctuation mark we detect and then sum the amount and return that value
"""

df["Punctuation_Count"] = df["text"].apply(count_punctuation)
"""Again really basic creation of a column from the text column and then applying that intial function that we have made"""

df["Punctuation_Density"] = (
    np.divide(df["Punctuation_Count"], df["Word_Count"])
    .fillna(0)
    .replace([np.inf, -np.inf], 0)
)
"""
Using numpy we first divide the the new Punctuation_Count we have created by the Word_Count column which we have also created.
We then fill any values that are not a number with 0. We then store these values in a new column Punctuation_Density
"""

df["Avg_Word_Length"] = df["text"].apply(
    lambda x: (
        np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
)
"""
This is again another column we created for analysis for a out heatmap. We again using functions like lambda to make sure that
iterate through every row in the text column and then we covert everything to strings and create a list of the length of each word
Then finally to make sure the logic is correct if there are words we calculate the mean and if there are no words just return the value
of 0. Then store all these values in the newly created column of Avg_Word_Length
"""

"""
We ultimately believed that all off these metrics were important to compare between the Fake News and Real News Datasets
for a multitude of reasons:
- Comparing the Punctuation could indicate if Fake News uses more ! and ?. While Real News doesn't
- Similarly Punctuation Density could indicate if the sentences for a certain Dataset are shorter
- Punctuation Count just gives an overall view of both datasets
- Polarity and Subjectivity are useful functions built into TextBlob that are valuable metrics to measure the pieces against each other
"""

y = df["label"]
# Relates back to the assigning of fake as 0 and real as 1 from earlier

X = df[
    ["Word_Count", "Subjectivity", "Polarity", "Punctuation_Density", "Avg_Word_Length"]
]
# Uses the columns that we created to now create the heatmap

correlation_df = pd.concat([X, y], axis=1)
"""
Uses the same pandas concat feature to combine these very columns to gether, but setting their axis to be 1 so that
columns will add next to each other rather than adding on top of each other
"""

correlation_matrix = correlation_df.corr()
"""
This is super duper useful because it now takes all these brand new columns that we created and calculates the correlation
between them without us having to do any of the work.
"""

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Correlation Coefficient"},
)
plt.title("Correlation Heatmap of Features in Fake and Real News Dataset")
plt.xticks(rotation=45, ha="right")  #
plt.yticks(rotation=0)
plt.savefig("correlation_heatmap.png")
plt.show()

"""
Essentially plotting of the heatmap correlation using seaborn and matplotlib
And we have succesfully created our heatmap correlation for our fake and real news datasets
"""

# Codebase and comments done by Hayden
