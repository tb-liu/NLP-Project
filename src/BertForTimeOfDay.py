# this try to use bert for time of day classification
# just a wild thought that would the review imply the time 
# the review was posted
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset, DatasetDict

time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning', time_dict['Morning']
    elif 12 <= hour < 17:
        return 'Afternoon', time_dict['Afternoon']
    elif 17 <= hour < 21:
        return 'Evening', time_dict['Evening']
    else:
        return 'Night', time_dict['Night']

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

df = pd.read_csv('./data/reviews_with_temporal_info.csv')
df = df.sample(10000)
# Convert the 'date' column to datetime type explicitly
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # 'coerce' will set invalid parsing as NaT

# Apply the function and create a new temporary column for the tuples
df['time_and_label'] = df['date'].dt.hour.apply(lambda x: categorize_time_of_day(x) if pd.notna(x) else (None, None))

# Split the tuple into two separate columns
df[['time_of_day', 'label']] = pd.DataFrame(df['time_and_label'].tolist(), index=df.index)

# drop the temporary column
df.drop('time_and_label', axis=1, inplace=True)

# Display the updated DataFrame to verify the new columns
print(df[['date', 'time_of_day', 'label']].head())

# base model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(time_dict))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# data preperation
hf_dataset = Dataset.from_pandas(df)
hf_dataset = hf_dataset.map(tokenize_function, batched=True)
hf_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_val_split = hf_dataset.train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_val_split['train'],
    'val': train_val_split['test']
})

training_args = TrainingArguments(
    output_dir='./model/results/TOD',          # Where to store the output files
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Weight decay for regularization
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
results = trainer.evaluate()
trainer.save_model('./model/time_of_day_prediction')
print(results)
# lol 
# {'eval_loss': 1.2533844709396362, 'eval_accuracy': 0.423, 'eval_f1': 0.14862965565706254, 
# 'eval_precision': 0.10575, 'eval_recall': 0.25, 'eval_runtime': 52.3112, 
# 'eval_samples_per_second': 38.233, 'eval_steps_per_second': 2.39, 'epoch': 2.0}
# does not seems working, no implicit info of when the review was posted
# Those review are data with temporal info, I would imagine the reviews without 
# temporal info will be worse. 