import pandas as pd
from transformers import pipeline, BertForTokenClassification, AutoTokenizer
import ast
import re

def normalize_text(phrase):
    normalized_phrase = re.sub(r'[^a-z0-9]', '', phrase.lower())
    return (normalized_phrase, phrase)

# Function to extract phrases from predictions
def extract_phrases(predictions):
    phrases = []
    current_phrase = []
    for prediction in predictions:
        for token_info in prediction:
            token = token_info['word'].replace('##', '')
            label = label_names[token_info['entity']]

            # Start a new phrase if a B-TEMP label is found
            if label == 'B-TEMP':
                if current_phrase:
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                current_phrase.append(token)
            elif label == 'I-TEMP':
                current_phrase.append(token)
    
    # Append the last phrase if any
    if current_phrase:
        phrases.append(" ".join(current_phrase))
    return phrases

# Function to compare extracted phrases with ground truths
def compare_phrases(predictions, ground_truths, texts, max_length = 500):
    results = []
    for prediction, ground_truth, text in zip(predictions, ground_truths, texts):
        extracted_phrases = [normalize_text(phrase) for phrase in extract_phrases([prediction])]
        extracted_set = {norm for norm, _ in extracted_phrases}
        original_phrases = {orig for _ , orig in extracted_phrases}

        # Normalize ground truth phrases and retain originals
        ground_truth_normalized = [normalize_text(gt) for gt in ground_truth]
        ground_truth_set = {norm for norm, _ in ground_truth_normalized}
        ground_truth_original = {orig for _ , orig in ground_truth_normalized}

        # Check if sets of phrases are different
        if set(extracted_set) != set(ground_truth_set):
            display_text = text if len(text) <= max_length else "Text is too long to display."
            results.append({
                "Original Sentence": display_text,
                "Extracted Phrases": original_phrases,
                "Ground Truth": ground_truth_original,
            })
    return results

label_names = {'LABEL_0': 'O', 'LABEL_1': 'B-TEMP', 'LABEL_2': 'I-TEMP'}
special_tokens = {'[CLS]', '[SEP]', '[PAD]'}
tag_to_ID = {'O': 0, 'B-TEMP': 1, 'I-TEMP': 2, '[CLS]': -100, '[SEP]': -100, '[PAD]': -100}

df = pd.read_csv('./data/reviews_with_temporal_info.csv')
df = df.sample(100)
df['temporal_info'] = df['temporal_info'].apply(ast.literal_eval)

# Load your trained model
model_path = './model/temporalInfo'
model = BertForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create a token classification pipeline
nlp_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

print(df['text'].head())
# Get predictions
predictions = nlp_pipeline(df['text'].to_list())


mismatches = compare_phrases(predictions, df['temporal_info'], df['text'])
for mismatch in mismatches:
    print(f"Original Sentence: {mismatch['Original Sentence']}")
    print(f"Mismatch Found - Extracted Phrases: {mismatch['Extracted Phrases']} | Ground Truth: {mismatch['Ground Truth']}")