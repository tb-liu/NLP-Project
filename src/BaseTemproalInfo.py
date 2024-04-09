# This file is to extract temporal info using other library, 
# The output can be used as training set for Bert fine-tuning
import pandas as pd
import spacy

# Function to extract temporal entities from text
def extract_temporal_entities(text):
    doc = nlp(text)
    temporal_entities = []
    for ent in doc.ents:
        if ent.label_ in ["TIME", "DATE"]:
            temporal_entities.append(ent.text)
    return temporal_entities

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")
review_df = pd.read_json('./data/yelp_academic_dataset_review.json', lines=True)

# Sample a subset of the data, e.g., 1000 reviews
sample_size = 1000  # You can adjust this number based on your needs
sampled_df = review_df.sample(n=sample_size, random_state=1)

# Extract temporal information for each review in the sample
sampled_df['temporal_info'] = sampled_df['text'].apply(extract_temporal_entities)

# Save the extracted data to a new CSV file
sampled_df[['text', 'temporal_info']].to_csv("./data/sampled_yelp_reviews_with_temporal_info.csv", index=False)