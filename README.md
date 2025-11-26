# Fake News vs Real News
This is a collaborative project built by two UCLA students Hayden Samala (Mathematics of Computation) & Kavin Ramesh (Statistics & Data Science). 

We developed several basic data visualizations, logistic regression charts, and a full fledged Natural Language Processing Model all from a kaggle dataset to help classify real vs fake news.

This project aims to build an NLP model capable of analyzing a news article and correctly prediciting if an article is fake news or has legitimate information.

# Basic Visualizations
We created a series of visualizations to track correlation in word count, subjectivity, polarity, punctuation density, and average word length between the fake news articles and real news articles. Additionally, we created feature importance plot to pick which features were most important in detecting what is real or fake news. Furthermore, we created basic word clouds and a density graph comparing word density between the real and fake articles.

## Heatmap Correlation:
<img width="1000" height="800" alt="Image" src="https://github.com/user-attachments/assets/8912bb4d-19da-4f2e-91dc-cbea5f465f70" />

## Feature Importance Plot:
<img width="1000" height="800" alt="Image" src="https://github.com/user-attachments/assets/f66818db-1eb3-4636-b247-3291bfdf2563" />

## Word Count Density:
<img width="1200" height="600" alt="Image" src="https://github.com/user-attachments/assets/53353169-7da8-4530-9ebc-0061e335cf84" />

## Word Clouds of Fake & Real News
<img width="1600" height="800" alt="Image" src="https://github.com/user-attachments/assets/a267ca9a-2d97-4371-9c65-3f1bb0e69fcd" />

# Logistic Regression Analysis:
We created a series of charts based off of logistic regression principles. 

## Confusion Matrix:
<img width="800" height="600" alt="Image" src="https://github.com/user-attachments/assets/697b9d86-1f7f-4d99-a93e-8c9ffe4069c4" />

## Distribution of Prediction Probabilites:
<img width="800" height="500" alt="Image" src="https://github.com/user-attachments/assets/33dc33d0-d219-400f-93b6-302b664d8fc8" />

## Probabilities by Label:
<img width="800" height="500" alt="Image" src="https://github.com/user-attachments/assets/8816537d-a32c-4aeb-83c6-20d5d22a6afd" />

## Prediction Distribution:
<img width="800" height="500" alt="Image" src="https://github.com/user-attachments/assets/af07708a-1989-42bd-b570-bdcb4bf43567" />

# Natural Language Model Building Process:
We intially worked by building out using a pretrained roberta-base model which then used and existing kaggle dataset containing real news and fake news from the period of 2015 to 2017. 

## Features:
- News Article Classification
- Text Preprocessing Pipeline
- Model Training + Evaluation
- Explainability Tools
- Dataset Integration

## Results:

When we eventually ran our first NLP model we received the following results:
- eval_loss: 0.0125
- eval_accuracy: 0.9986
- eval_f1_score: 0.9986
- eval_precision: 0.9986
- eval_recall: 0.9986
- eval_runtime: 97.5257
- eval_samples_per_second: 45.3930
- eval_steps_per_second: 5.6810
- epoch: 3.0000

These results were incredibly concerning because it indicated that the model was almost too accurate. This then indicated that there were several issues within our datasets. After performing verification tests we saw that every single row within the True.csv had the associated "REUTERS" tag to it, adding to the inflated accuracy measurements. Additionally, there were subject differences between the Fake.csv and True.csv with them both classifying articles differently. This then lead to us dropping the subject column for our NLP model training. 
As such we then made several modifications to our codebase to account for this:
- At the start of the codebase we added a function to remove all the tags of <REUTERS> from our code and then created a brand new column to store all that new text information and deleted the old one
- To optimize the training we changed several of our key metrics such as changing our eval_steps

This was then the results of our second NLP model testing:
- eval_loss: 0.0259
- eval_accuracy: 0.9964
- eval_f1_score: 0.9963
- eval_precision: 0.9958
- eval_recall: 0.9967
- eval_runtime: 92.2295
- eval_samples_per_second: 48.0000
- eval_steps_per_second: 6.0070
- epoch: 3.0000

After performing data cleaning and backtesting, we still produced a model of incredibly high accuracy. This indicates that even though there were issues with Reuters tag and with our training arguments due to Roberta-bases incredible fitting towards this type of work it was highly successful in predicting fake news vs real news.

## Key Findings & Realizations:
Roberta-base model is a obviously a pretrained model which has been trained on millions of lines of sentences to build general speech recongnition. However, without optimizing it for our needs we essentially overfitting our data with the model. As such we learned that we ashould be trained on a downstream task to better handle and be able to perform what we want it to do, which is to predict real vs fake news. We also learned to look to better clean our data before creation of any models or analysis as it could interfere with our results.

## Dataset:
This is the dataset we used to build out all which we have done:

https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

## Key Notes:
Kavin had issues pushing code to github so almost all pushes were done by Hayden. However, work was still down equally between the two of us.
Additionally, all code blocks should have extensive comments explaining our reasoning

