import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

fake_news = pd.read_csv('Fake.csv')
real_news = pd.read_csv('True.csv')

print(fake_news.head())
print(real_news.head())
print(fake_news.columns)
print(real_news.columns)