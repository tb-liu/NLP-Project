# test.py
import pandas as pd
import numpy as np
import json
import joblib

# Load the model
model = joblib.load('./models/attr_model.xgb')

# Load validation data
X_valid = pd.read_csv('./data/test.csv')

# Assuming 'business_id' and 'highProbAttr' are columns in your DataFrame
X_valid_encoded = pd.get_dummies(X_valid.drop(['highProbAttr', 'business_id'], axis=1), drop_first=True)

# Make predictions
predictions = model.predict(X_valid_encoded)
print("Sample Predictions:", predictions[:10])

# Find the most and least likely attribute presence
mostLikelyAttr = np.max(predictions)
leastLikelyAttr = np.min(predictions)

# Find the businesses corresponding to the highest and lowest predictions
highAttrIndex = np.where(predictions == mostLikelyAttr)[0]
lowAttrIndex = np.where(predictions == leastLikelyAttr)[0]

highAttrRow = X_valid.iloc[highAttrIndex]
lowAttrRow = X_valid.iloc[lowAttrIndex]

print("Business ID most likely to have the attribute:", highAttrRow['business_id'].values)
print("Business ID least likely to have the attribute:", lowAttrRow['business_id'].values)

# Optionally, read reviews for demonstrative purposes
mostLikelyAttrReviews = pd.DataFrame(columns=['text'])
leastLikelyAttrReviews = pd.DataFrame(columns=['text'])

with open("./data/yelp_academic_dataset_review.json", encoding="utf-8") as reviews:
    for review in reviews:
        review = reviews.readline()
        if not review:
            break
        reviewData = json.loads(review)
        
        if reviewData['business_id'] in highAttrRow['business_id'].values:
            mostLikelyAttrReviews = mostLikelyAttrReviews._append({'text': reviewData['text']}, ignore_index=True)
        
        if reviewData['business_id'] in lowAttrRow['business_id'].values:
            leastLikelyAttrReviews = leastLikelyAttrReviews._append({'text': reviewData['text']}, ignore_index=True)

# Displaying sample reviews if they are loaded
if not mostLikelyAttrReviews.empty:
    pd.set_option('display.max_colwidth', None)
    print("Most likely to have the attribute, sample reviews:\n", mostLikelyAttrReviews.sample(n=2))
if not leastLikelyAttrReviews.empty:
    print("Least likely to have the attribute, sample reviews:\n", leastLikelyAttrReviews.sample(n=2))
