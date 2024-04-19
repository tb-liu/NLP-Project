import pandas as pd
from transformers import AutoTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
import evaluate
import numpy as np
import ast

metric = evaluate.load("seqeval")
label_names = {0: 'O', 1: 'B-TEMP', 2: 'I-TEMP'}
special_tokens = {'[CLS]', '[SEP]', '[PAD]'}
tag_to_ID = {'O': 0, 'B-TEMP': 1, 'I-TEMP': 2, '[CLS]': -100, '[SEP]': -100, '[PAD]': -100}
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# Assign labels for temporal phrases using BIO scheme
def token_to_tag_id(tokens, tokenized_temporal_phrases, to_id = False):
    if to_id:
        labels = [0] * len(tokens)  # Numeric ID for 'O'
    else:
        labels = ['O'] * len(tokens)  # String label 'O'
    # keep the special tokens
    for i, token in enumerate(tokens):
        if token in special_tokens:
            labels[i] = token if not to_id else tag_to_ID[token]
    for phrase, phrase_tokens in tokenized_temporal_phrases.items():
        start_index = 0
        while start_index < len(tokens):
            # Search for the start of the phrase in tokens
            if phrase_tokens[0] in tokens[start_index:]:
                start_index = tokens.index(phrase_tokens[0], start_index)
            else:
                break

            # Check if the full phrase matches
            if tokens[start_index:start_index + len(phrase_tokens)] == phrase_tokens:
                # Label the first token as B-TEMP
                labels[start_index] = 'B-TEMP' if not to_id else tag_to_ID['B-TEMP']
                # Label subsequent tokens as I-TEMP
                for i in range(1, len(phrase_tokens)):
                    labels[start_index + i] = 'I-TEMP' if not to_id else tag_to_ID['I-TEMP']
                start_index += len(phrase_tokens)  # Move index past the end of the current phrase
            else:
                start_index += 1  # Increment index to keep searching
    return labels


def create_labels(text, temporal_words):
    # Tokenize text and include special tokens
    encoded_dict = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_attention_mask=True)
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Convert ids to tokens for alignment

    # Tokenize temporal phrases to understand how they might be split
    tokenized_temporal_phrases = {phrase: tokenizer.tokenize(phrase) for phrase in temporal_words}

    # Assign labels
    labels = token_to_tag_id(tokens, tokenized_temporal_phrases, to_id=True)
    return tokens, labels, attention_mask, input_ids

class TemporalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

df = pd.read_csv('./data/reviews_with_temporal_info.csv')
df = df.sample(300)

# Apply function to each row in DataFrame
# Convert the 'temporal_info' from string representation of a list to an actual list
df['temporal_info'] = df['temporal_info'].apply(ast.literal_eval)
df['processed'] = df.apply(lambda row: create_labels(row['text'], row['temporal_info']), axis=1)
df[['tokens', 'labels', 'attention_mask', 'input_ids']] = pd.DataFrame(df['processed'].tolist(), index=df.index)
# df['flat_labels'] = df['labels'].apply(flatten_labels)
print(df.head()[['processed', 'labels']])

# Prepare encodings and labels
encodings = {'input_ids': list(df['input_ids']), 'attention_mask': list(df['attention_mask'])}
labels = list(df['labels'])

labels_tensor = torch.tensor(labels)  # Convert labels list to a tensor
# Flatten the labels tensor
labels_flat = labels_tensor.view(-1)
labels_flat = labels_flat[labels_flat != -100]
# Calculate the number of occurrences of each class
class_counts = torch.bincount(labels_flat, minlength=3)  # Assuming three classes

# Calculate weights: Number of samples / (number of classes * number of samples for each class)
total_samples = labels_tensor.size(0)
num_classes = 3
weights = total_samples / (class_counts * num_classes)

print("Class weights:", weights)
weights = weights.to(device)
# Create dataset
dataset = TemporalDataset(encodings, labels)
# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_names))  # 3 labels: 'O','I-TEMP' and 'B-TEMP'

training_args = TrainingArguments(
    output_dir='./model/results',          # Where to store the output files
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=64,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Weight decay for regularization
    evaluation_strategy="epoch",
    save_strategy="epoch"
)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Resize labels and logits for CrossEntropyLoss
        loss_fct = CrossEntropyLoss(weight=weights.float())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
results = trainer.evaluate()
trainer.save_model('./model/temporalInfo')
print(results)
