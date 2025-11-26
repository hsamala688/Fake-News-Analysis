# Fake-New-Analysis-NLP-Model
This is a collaborative project built by two UCLA students Hayden Samala (Mathematics of Computation) & Kavin Ramesh (Statistics & Data Science) to classify online news articles as real of fake using modern NLP and machine-learning techniques.

This project aims to build an NLP model capable of analyzing a news article and correctly prediciting if an article is fake news or has legitimate information.

# Basic Visualizations
We created a series of visualizations to track correlation in word count, subjectivity, polarity, punctuation density, and average word length between the fake news articles and real news articles. Additionally, we created feature importance plot to pick which features were most important in detecting what is real or fake news. Furthermore, we created basic word clouds and a density graph comparing word density between the real and fake articles.

## Heatmap Correlation:
<img width="1000" height="800" alt="Image" src="https://github.com/user-attachments/assets/e506e549-c1dc-4fa5-a9a7-f72fa1f3816d" />

## Feature Importance Plot:
<img width="1000" height="800" alt="Image" src="https://github.com/user-attachments/assets/f66818db-1eb3-4636-b247-3291bfdf2563" />

## Word Count Density:
<img width="1200" height="600" alt="Image" src="https://github.com/user-attachments/assets/53353169-7da8-4530-9ebc-0061e335cf84" />

## Word Clouds of Fake & Real News
<img width="1600" height="800" alt="Image" src="https://github.com/user-attachments/assets/a267ca9a-2d97-4371-9c65-3f1bb0e69fcd" />

# Natural Language Model Building Process:
We intially worked by building out using a pretrained roberta-base model which then used and existing kaggle dataset containing real news and fake news from the period of 2015 to 2017. 

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



## Key Findings & Realizations:
Roberta-base model is a obviously a pretrained model which has been trained on millions of lines of sentences to build general speech recongnition. However, without optimizing it for our needs we essentially overfitting our data with the model. As such we learned that we ashould be trained on a downstream task to better handle and be able to perform what we want it to do, which is to predict real vs fake news. We also learned to look to better clean our data before creation of any models or analysis as it could interfere with our results.

# Features:
- News Article Classification
- Text Preprocessing Pipeline
- Model Training + Evaluation
- Explainability Tools
- Dataset Integration

