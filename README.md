# Fake-New-Analysis-NLP-Model

This is a collaborative project built by two UCLA students Kavin Ramesh (Statistics & Data Science ) & Hayden Samala (Mathematics of Computation) to classify online news articles as real of fake using modern NLP and machine-learning techniques.

This project aims to build an NLP model capable of analyzing a news article and correctly prediciting if an article is fake news or has legitimate information.

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

