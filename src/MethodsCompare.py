import pandas as pd
from transformers import pipeline, BertForTokenClassification, AutoTokenizer
import ast
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
def compare_phrases(predictions, ground_truths):
    results = []
    for prediction, ground_truth in zip(predictions, ground_truths):
        # Extract phrases and convert them to lowercase for case-insensitive comparison
        extracted_phrases = [phrase.lower() for phrase in extract_phrases([prediction])]
        ground_truth_set = [gt.lower() for gt in ground_truth]  # Assuming ground_truth is a list of strings

        # Convert lists to sets for comparison to identify mismatches
        extracted_set = set(extracted_phrases)
        ground_truth_set = set(ground_truth_set)
        # Check if sets of phrases are different
        if set(extracted_set) != set(ground_truth_set):
            results.append({
                "Extracted Phrases": extracted_phrases,
                "Ground Truth": ground_truth
            })
    return results

label_names = {'LABEL_0': 'O', 'LABEL_1': 'B-TEMP', 'LABEL_2': 'I-TEMP'}
special_tokens = {'[CLS]', '[SEP]', '[PAD]'}
tag_to_ID = {'O': 0, 'B-TEMP': 1, 'I-TEMP': 2, '[CLS]': -100, '[SEP]': -100, '[PAD]': -100}

df = pd.read_csv('./data/reviews_with_temporal_info.csv')
df = df.sample(10)
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


# Extract and print phrases for each text
# for i in range(0, len(predictions)):
#     phrases = extract_phrases([prediction])  # Note: wrapping prediction in a list

#     print(f"Extracted Phrases: {phrases}")

mismatches = compare_phrases(predictions, df['temporal_info'])
for mismatch in mismatches:
    print(f"Mismatch Found - Extracted Phrases: {mismatch['Extracted Phrases']} | Ground Truth: {mismatch['Ground Truth']}")