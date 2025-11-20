import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import torch
import sklearn
from transformers import AutoTokenizer
from transformers import TrainingArguments
from nltk.corpus import stopwords #stopwords for english
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words_english = stopwords.words('english')

fake_news_df = pd.read_csv('Fake.csv')
real_news_df = pd.read_csv('True.csv')

fake_news_df['label'] = 0
real_news_df['label'] = 1

#Basically this is the whole patch to clean the 'Reuters' tag from the real news dataset. Help model Traning
pattern_to_remove = r'^[A-Z/,\s]+\s*\(REUTERS\)\s*-\s*'
real_news_df['text_clean'] = real_news_df['text'].astype(str).str.replace(
    pattern_to_remove, 
    '', 
    regex=True, 
    flags=re.IGNORECASE
)
real_news_df.drop('text', axis=1, inplace=True)
real_news_df.rename(columns={'text_clean': 'text'}, inplace=True)

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
        .apply(lambda x: x.split())
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
        .apply(lambda x: x.split())
        .apply(lambda x: [word for word in x if word not in stop_words_english])
        .apply(lambda x: ' '.join(x))
    )

real_news_df = real_news_df[real_news_df['text'].str.len() > 0]
real_news_df.to_csv("True_Cleaned.csv", index=False)
'''print(fake_news_df['subject'].value_counts())
print(real_news_df['subject'].value_counts())'''

#Step 2: Finding Patterns in Words of the Fake News Dataset

from sklearn.utils import class_weight
from transformers import Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, RobertaForSequenceClassification, TrainingArguments, IntervalStrategy
from transformers import EarlyStoppingCallback
from datasets import Dataset

model_name = "roberta-base"
max_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_num_fn = len(fake_news_df)
total_num_rn = len(real_news_df)
all_labels = np.concatenate([np.zeros(total_num_fn), np.ones(total_num_rn)])

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

print(f"Calculated Class Weights:")
print(f"Weight for Fake News (Class 0): {class_weights_tensor[0]:.4f}") # 0.9686
print(f"Weight for Real News (Class 1): {class_weights_tensor[1]:.4f}") # 1.0335

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights_device = class_weights_tensor.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights_device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# The WeightedLossTrainer can now be used to train the model with class-weighted loss.
# This helps to address class imbalance during training.

training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy=IntervalStrategy.EPOCH,
    eval_steps = 450, #Calculated: Total Dataset Size = 44898*0.81 = 36380/Batch Size: 8 =  4545.9225/10 = ~450
    num_train_epochs= 10,
    load_best_model_at_end=True,
    metric_for_best_model="f1_score", #it balances precision and recall, especially important with your slightly imbalanced dataset
    greater_is_better=True, #We want the f1_score to be as high as possible
    weight_decay=0.01,
)

df = pd.concat([fake_news_df, real_news_df], ignore_index=True)
df.drop(['subject'], axis=1, inplace=True) #another part of the patch because of the descrepency in subject labels, discussed in Verification_of_Data_Testing.py

df_train_val, df_test = train_test_split(
    df, 
    test_size=0.10,          # Reserve 10% for the final Test Set
    random_state=42,         # Ensure reproducibility
    stratify=df['label']     # IMPORTANT: Keep Fake/Real proportions balanced
)

df_train, df_val = train_test_split(
    df_train_val, 
    test_size=(0.10 / 0.90), # Calculate the test_size relative to the new subset
    random_state=42, 
    stratify=df_train_val['label']
)

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
test_dataset = Dataset.from_pandas(df_test)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=max_length)
train_dataset = train_dataset.map(tokenize, batched = True)
val_dataset = val_dataset.map(tokenize, batched = True)
test_dataset = test_dataset.map(tokenize, batched = True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type = "torch", columns = ['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type = "torch", columns = ['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type = "torch", columns = ['input_ids', 'attention_mask', 'labels'])    

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    true_labels = p.label_ids
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds, average='binary', zero_division=0)
    recall = recall_score(true_labels, preds, average='binary', zero_division=0)
    f1 = f1_score(true_labels, preds, average='binary', zero_division=0)
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        # Optionally add components of the confusion matrix for deep analysis:
        # 'true_positives': TP,
        # 'false_negatives': FN, 
    }


trainer = WeightedLossTrainer(
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2),
    args=training_args,
    compute_metrics = compute_metrics,
    tokenizer=tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset  
    )

trainer.train()

print("Evaluating the Model")
evaluation_results = trainer.evaluate(eval_dataset=val_dataset)

for key, value in evaluation_results.items():
    print(f"{key}: {value:.4f}")

#Intial Training Results:
'''
eval_loss: 0.0125
eval_accuracy: 0.9986
eval_f1_score: 0.9986
eval_precision: 0.9986
eval_recall: 0.9986
eval_runtime: 97.5257
eval_samples_per_second: 45.3930
eval_steps_per_second: 5.6810
epoch: 3.0000
'''