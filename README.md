# Fake-New-Analysis-NLP-Model
This is a collaborative project built by two UCLA students Kavin Ramesh (Statistics & Data Science ) & Hayden Samala (Mathematics of Computation) to classify online news articles as real of fake using modern NLP and machine-learning techniques.

This project aims to build an NLP model capable of analyzing a news article and correctly prediciting if an article is fake news or has legitimate information.

# Basic Visualizations
We created a series of visualizations to track correlation in word count, subjectivity, polarity, punctuation density, and average word length between the fake news articles and real news articles. Additionally, we created feature importance plot to pick which features were most important in detecting what is real or fake news. Furthermore, we created basic word clouds and a density graph comparing word density between the real and fake articles.

Heatmap Correlation:
<img width="1000" height="800" alt="Image" src="https://github.com/user-attachments/assets/e506e549-c1dc-4fa5-a9a7-f72fa1f3816d" />

Feature Importance Plot:
<img width="1000" height="600" alt="Image" src="https://github.com/user-attachments/assets/4308b521-9237-414c-82d4-232afc0fba4f" />

Word Count Density:
<img width="1600" height="800" alt="Image" src="https://github.com/user-attachments/assets/a267ca9a-2d97-4371-9c65-3f1bb0e69fcd" />

Word Clouds of Fake & Real News
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

This was then the results of our second NLP model testing:

# Features:
- News Article Classification
- Text Preprocessing Pipeline
- Model Training + Evaluation
- Explainability Tools
- Dataset Integration

