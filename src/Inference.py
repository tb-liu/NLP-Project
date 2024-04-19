from transformers import pipeline, BertForTokenClassification, AutoTokenizer

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

# Print the predictions
for prediction in predictions:
    print(prediction)