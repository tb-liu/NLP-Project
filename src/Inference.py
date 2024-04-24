from transformers import pipeline, BertForTokenClassification, AutoTokenizer
label_names = {'LABEL_0': 'O', 'LABEL_1': 'B-TEMP', 'LABEL_2': 'I-TEMP'}
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

# Load your trained model
model_path = './model/temporalInfo'
model = BertForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Create a token classification pipeline
nlp_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)

# Example text
text = "I visited the place last June. They open daily from 9 AM to 5 PM."

# Get predictions
predictions = nlp_pipeline(text)
predictions = extract_phrases([predictions])
# Print the predictions
for prediction in predictions:
    print(prediction)