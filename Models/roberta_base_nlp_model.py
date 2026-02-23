import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import torch
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
'''
Essentially this is our part of normalizing and cleaning all text data to make sure that we are ready to go for model creation
You will notice that we still have subject in here, but after we realized our mistake we allow it to continue being processed in our cleaning
but just dropped later on
'''

for col in fn_text_columns:
    fake_news_df[col] = (
        fake_news_df[col]
        .astype(str)
        .str.lower()
        .str.replace("[^a-zA-Z ]", " ", regex=True)  # removes apostrophes too
    )
'''
Using a for loop to clean through all the columns of the dataframe that we created and making sure everything is a string, everything is lowercase
and that apostrophes are removed because they are irrelevant to the later tokenization that we perform
'''

for col in fn_text_columns:
    fake_news_df[col] = (
        fake_news_df[col]
        .apply(lambda x: x.split())
        .apply(lambda x: [word for word in x if word not in stop_words_english])
        .apply(lambda x: ' '.join(x))
    )
'''
This second part performs so more deep cleaning based on our intial set up. Now it goes row through and splits words from
that intial string into seperate entities. It then using the nltk packages of stop words filters out really basic words such as 
the, it, as, and, a and removes them. Then we combine everything back into one string.
This is done so that we can perform proper cleaning for our NLP model. We then repeated this same process for the real news dataset.
'''

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
'''
These intial print statemnts were used to try and see what came up as the subject classifications within the subject columns
'''

#Step 2: Finding Patterns in Words of the Fake News Dataset

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, RobertaForSequenceClassification, TrainingArguments, IntervalStrategy
from transformers import Trainer
from datasets import Dataset

'''
These lines above are ultimately the backbone of our NLP Model. We used sklearn and class_weight because of the fact that
the number of rows between our real and fake datasets is uneven, requiring the class_weight package to be able to manage that.
Train_test_split was to establish how we would then split our training data, which we ultimately did 90 to 10.
Accuracy_score, Precision_Score, recall_score, f1_score are all used to measure the accuracy of the model against common metrics
Then the transfomers libraries are used to then build our robertabase model. Like the autotokenizer makes sure to load the correct
number of tokens for our model, the sequenceclassification is to make sure its set up for fake news detection.
Training Arguments is used so that we could optimize and specifically assign our own arguments based on our needs.
InvtervalStrategy was used to manage the number of steps in training (epoch every N steps) so as not to overwhelm our macbooks
Trainer simple handles the entire loop of training for us so that we dont have to manage that
Datasets was simple used to convert all of our Pandas dataframs to be readable by transformers
'''

'''
We ultimately decided to use the pretrained robertabase model for a number of reasons:
- It was proven model for fake news detection
- It can pick up subtle stylistic choices in writing due to what it was intially trained on
- Though it runs slower it can also produce a far higher accuracy
'''

model_name = "roberta-base"
max_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_name)
'''
These lines load the roberta-base model, set the max sequence of tokens to 512 and sets the tokenizer to be that said model
'''


total_num_fn = len(fake_news_df)
total_num_rn = len(real_news_df)
all_labels = np.concatenate([np.zeros(total_num_fn), np.ones(total_num_rn)])
'''
The creates the 0 and 1 for the real and fake news and then combines them all into one dataset
'''

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)
'''
Class weights handles the imbalanced data that we have using the formula: n_samples / (n_classes * n_samples_per_class)
Producing the wieghts that you can see later on
This makes the model ultimately pay attention to the smaller dataset (real news) a little bit more because there are simple
less test cases for it
'''
class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

'''
print(f"Calculated Class Weights:")
print(f"Weight for Fake News (Class 0): {class_weights_tensor[0]:.4f}")  -> 0.9686
print(f"Weight for Real News (Class 1): {class_weights_tensor[1]:.4f}")  -> 1.0335
'''

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

'''
For our Trainer we overrided the intial trainer values built in to accomadate for the custom weighting that we had
to perform because of our datasets. It also moves all the weights to the same as device to maintain CPU and GPU compaibility
It then finally applies the class weights during loss classification to make sure that our metrics come out correctly
'''

training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy=IntervalStrategy.EPOCH,
    eval_steps = 450, #Calculated: Total Dataset Size = 44898*0.81 = 36380/Batch Size: 8 =  4545.9225/10 = ~450
    num_train_epochs= 3,
    load_best_model_at_end=True,
    metric_for_best_model="f1_score", #it balances precision and recall, especially important with your slightly imbalanced dataset
    greater_is_better=True, #We want the f1_score to be as high as possible
    weight_decay=0.01,
)
'''
We kept our learing rate at 1e-5 because that is standard for pretrained transfomers
We had to up the number of epochs because as you can in comparison to our previous epoch to make sure that with the changes
that we made our accuracy doesn't suffer
Weigted Decay was to prevent the overfiitng that we suffering when we were intially running our model
'''

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

'''
Essentially the splitting of our data into training and testing with stratification to ensure that the class distribution is 
even
'''

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
test_dataset = Dataset.from_pandas(df_test)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=max_length)
train_dataset = train_dataset.map(tokenize, batched = True)
val_dataset = val_dataset.map(tokenize, batched = True)
test_dataset = test_dataset.map(tokenize, batched = True)

'''
Converts the text into tokens that roberta can understand, adds padding tokens to make the sequences all the same length
and then cuts and sequence longer than 512 and finally processes multiple entries at the same time to work faster
'''

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

'''
we have to rename everything to labels to be compatible for the TrainerAPI
'''

train_dataset.set_format(type = "torch", columns = ['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type = "torch", columns = ['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type = "torch", columns = ['input_ids', 'attention_mask', 'labels'])   

'''
converts everything to torch sensors for processing
'''

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

'''
Accuracy: Overall Correctness of our Model
f1_score: Balance between precision and recall
recall: how much fake news model caught
precision: of predicted fake news how much was actually fake
'''


trainer = WeightedLossTrainer(
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2),
    args=training_args,
    compute_metrics = compute_metrics,
    tokenizer=tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset  
    )

trainer.train()


#intializes our trainer to finally be run
#loads the pre-trained roberta to train our data


print("Evaluating the Model")
evaluation_results = trainer.evaluate(eval_dataset=val_dataset)

for key, value in evaluation_results.items():
    print(f"{key}: {value:.4f}")


#our evaluation metrics get printed at the end


#Intial Training Results:

#eval_loss: 0.0125
#eval_accuracy: 0.9986
#eval_f1_score: 0.9986
#eval_precision: 0.9986
#eval_recall: 0.9986
#eval_runtime: 97.5257
#eval_samples_per_second: 45.3930
#eval_steps_per_second: 5.6810
#epoch: 10.0000

#Second round of Training
#After patching and adding more robust data cleaning
#eval_loss: 0.0259
#eval_accuracy: 0.9964
#eval_f1_score: 0.9963
#eval_precision: 0.9958
#eval_recall: 0.9967
#eval_runtime: 92.2295
#eval_samples_per_second: 48.0000
#eval_steps_per_second: 6.0070
#epoch: 10.0000

#Third round of Training
#Using Google Colab and setting Epoch to 3
#eval_loss: 0.0293
#eval_accuracy: 0.9964
#eval_f1_score: 0.9963
#eval_precision: 0.9963
#eval_recall: 0.9963
#eval_runtime: 30.5524
#eval_samples_per_second: 144.8990
#eval_steps_per_second: 18.1330
#epoch: 3.0000

#Codebase built by Hayden and Kavin, Comments and documentation done by Hayden