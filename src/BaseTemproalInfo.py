# Qiaochu Liu
# This file is to extract temporal info using other library, 
# The output can be used as training set for Bert fine-tuning
import pandas as pd
import spacy
from tqdm import tqdm
import json
import math
import os
# Function to extract temporal entities from text
def extract_temporal_entities(text, nlp):
    doc = nlp(text)
    temporal_entities = []
    for ent in doc.ents:
        if ent.label_ in ["TIME", "DATE"]:
            temporal_entities.append(ent.text)
    return temporal_entities


def split_json_file(source_file_path, output_directory, num):
    # Determine the total number of lines (records) in the file
    with open(source_file_path, 'r') as source_file:
        total_records = sum(1 for line in source_file)
    
    records_per_file = math.ceil(total_records / num)

    with open(source_file_path, 'r') as source_file:
        current_file_index = 0
        current_record_count = 0
        output_file = None

        for line in tqdm(source_file, total=total_records, desc="Splitting file"):
            # Open a new file to write if necessary
            if current_record_count == 0 or current_record_count >= records_per_file:
                if output_file:
                    output_file.close()
                output_file_path = f'{output_directory}/part_{current_file_index + 1}.json'
                output_file = open(output_file_path, 'w')
                current_file_index += 1
                current_record_count = 0
            
            # Process and write the current line
            record = json.loads(line)
            json.dump(record, output_file)
            output_file.write('\n')  # Add a newline to separate JSON objects
            current_record_count += 1

        if output_file:
            output_file.close()

def load_excluded_business_ids(exclude_file_path):
    excluded_ids = set()
    with open(exclude_file_path, 'r') as file:
        for line in file:
            business = json.loads(line)
            excluded_ids.add(business['business_id'])
    return excluded_ids

def find_reviews_with_temporal_info(source_file_path, exclude_file_path, output_file_path, max_reviews, nlp):
    excluded_ids = load_excluded_business_ids(exclude_file_path)
    temporal_reviews = []

    with open(source_file_path, 'r') as file:
        progress_bar = tqdm(desc="Processing reviews", unit="review", total=max_reviews)
        for line in file:
            review = json.loads(line.strip())

            if review['business_id'] not in excluded_ids:
                temporal_entities = extract_temporal_entities(review['text'], nlp)

                if temporal_entities:
                    review['temporal_info'] = temporal_entities
                    temporal_reviews.append(review)
                    progress_bar.update(1)

                    if len(temporal_reviews) >= max_reviews:
                        break
        progress_bar.close()
    # Convert the list of reviews to a DataFrame and save to CSV
    df_temporal_reviews = pd.DataFrame(temporal_reviews)
    df_temporal_reviews.to_csv(output_file_path, index=False)

def main(argument = 0):
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")
    output_directory = './data/split_reviews'
    source_file_path = './data/yelp_academic_dataset_review.json'
    exclude_business = './data/filtered_yelp_business.json'

    num_splits = 8
    # if do split data
    if argument == 1:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        split_json_file(source_file_path, output_directory, num_splits)
    elif argument == -1: # read seperated files and process 
        combined_df = pd.DataFrame()  # Initialize an empty for final output
        for i in range(1, num_splits + 1):  # Loop through 
            file_path = os.path.join(output_directory, f'part_{i}.json')
            if os.path.exists(file_path):
                review_df = pd.read_json(file_path, lines=True)

            # Process each review in the DataFrame
            tqdm_desc = f"Processing {file_path}"
            with tqdm(total=len(review_df), desc=tqdm_desc) as pbar:
                # Wrapper function to update progress bar after each review
                def process_review(text):
                    result = extract_temporal_entities(text, nlp)
                    pbar.update(1)
                    return result

                review_df['temporal_info'] = review_df['text'].apply(process_review)
        

            # Filter out reviews that have no temporal information
            processed_df = review_df[review_df['temporal_info'].map(len) > 0]

            combined_df = pd.concat([combined_df, processed_df])
        # Save the combined data to a single CSV file
        combined_df.to_csv("./data/processed_reviews_with_temporal_info.csv", index=False)
    else:
        max_reviews = 30000
        find_reviews_with_temporal_info(source_file_path, exclude_business, './data/reviews_with_temporal_info.csv',max_reviews, nlp)


if __name__ == '__main__':
    main()