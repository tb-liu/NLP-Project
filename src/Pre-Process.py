import pandas as pd

# Load the datasets
business_df = pd.read_json('./data/yelp_academic_dataset_business.json', lines=True)
review_df = pd.read_json('./data/yelp_academic_dataset_review.json', lines=True)

# Aggregate the review counts per business_id in the review DataFrame
review_counts = review_df.groupby('business_id').size().reset_index(name='review_count_aggregated')

# Merge the business DataFrame with the aggregated review counts
merged_df = pd.merge(business_df, review_counts, on='business_id', how='left')

# Calculate the total of reviews and tips
merged_df['total_reviews_tips'] = merged_df['review_count'] + merged_df['review_count_aggregated']

# Filter out businesses with more than 50 reviews and tips combined
filtered_businesses = merged_df[merged_df['total_reviews_tips'] > 50]

# Count the number of attributes for each business and filter out businesses with more than 3 attributes
filtered_businesses = filtered_businesses[filtered_businesses['attributes'].apply(lambda x: len(x) if isinstance(x, dict) else 0) > 3]

# Filter reviews to include only those that match the business IDs in the filtered businesses
filtered_reviews = review_df[review_df['business_id'].isin(filtered_businesses['business_id'])]

# Aggregate reviews into a list per business
review_groups = filtered_reviews.groupby('business_id')['text'].apply(list).reset_index(name='reviews')

# Merge the aggregated reviews with the filtered businesses
final_df = pd.merge(filtered_businesses, review_groups, on='business_id', how='left')

# Save the combined DataFrame to a new JSON file
final_df.to_json('./data/filtered_yelp_business_reviews.json', orient='records', lines=True)
