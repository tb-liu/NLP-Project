import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Load review data
wifiAvailableData = []
wifiNotAvailableData = []

words = ["wifi", "internet", "Wi-Fi", "no wifi", "free wifi", "wireless", "no internet", "wireless connection"]
count = 0

# Load the review data
with open("./data/yelp_academic_dataset_review.json", encoding="utf-8") as reviews:
    for _ in range(100000):  # Limit to first 100,000 reviews for efficiency
        review = reviews.readline()
        if not review:
            break
        reviewData = json.loads(review)
        
        # if text contains one of the keywords related to wifi
        if any(word in reviewData['text'].lower() for word in words):
            if "no wifi" in reviewData['text'].lower() or "no internet" in reviewData['text'].lower():
                wifiNotAvailableData.append({'business_id': reviewData['business_id'], 'date': reviewData['date'], 'text': reviewData['text']})
            else:
                wifiAvailableData.append({'business_id': reviewData['business_id'], 'date': reviewData['date'], 'text': reviewData['text']})
        
        # Random selection for a balanced dataset
        elif np.random.randint(1, 180) == 1:
            wifiNotAvailableData.append({'business_id': reviewData['business_id'], 'date': reviewData['date'], 'text': reviewData['text']})
            count += 1

wifiAvailable = pd.DataFrame(wifiAvailableData, columns=['business_id', 'date', 'text'])
wifiNotAvailable = pd.DataFrame(wifiNotAvailableData, columns=['business_id', 'date', 'text'])

''' 

Preprocessing 

'''

# The preprocessing function remains largely the same, with changes to variable names and logic as needed
def formatDataFromID(business_ids, wifiAvailable=0):
    businessDataList = []
    
    with open("./data/yelp_academic_dataset_business.json", encoding="utf-8") as businesses:
        for business in businesses:
            businessData = json.loads(business)

            if businessData['business_id'] in business_ids:
                if businessData.get('categories') and ('Food' in businessData['categories'] or 'Restaurants' in businessData['categories']):
                    columnsToDrop = ["hours", "name", "address", "city", "state", "postal_code", "latitude", "longitude", "is_open", "BusinessParking", "GoodForMeal"]
                    for column in columnsToDrop:
                        businessData.pop(column, None)

                    categoryColumns = businessData["categories"].split(",")
                    for category in categoryColumns:
                        businessData['category_' + category.strip()] = 1
                    
                    businessData.pop('attributes', None)
                    businessData['wifiAvailable'] = wifiAvailable
                    businessDataList.append(businessData)
    
    X = pd.DataFrame(businessDataList)
    return X

wifiAvailableBusinesses = wifiAvailable['business_id'].tolist()
wifiNotAvailableBusinesses = wifiNotAvailable['business_id'].tolist()

X_wifiAvailable = formatDataFromID(wifiAvailableBusinesses, 1)
X_wifiNotAvailable = formatDataFromID(wifiNotAvailableBusinesses)

# Merge the data related to WiFi availability and non-availability
X = pd.concat([X_wifiAvailable, X_wifiNotAvailable], ignore_index=True)

# Dropping columns not relevant to WiFi prediction
X_clean = X.drop(['categories', 'category_Restaurants', 'business_id'], axis=1)

# Handling missing data and encoding categorical variables
cols_with_missing = [col for col in X.columns if X[col].isnull().sum() > X.shape[0]*0.9]
X_clean = X_clean.drop(cols_with_missing, axis=1)
X_clean = X_clean.fillna(0)
X_encoded = pd.get_dummies(X_clean, drop_first=True)

'''

Model Training

'''

# Separate features and target variable
y = X_encoded['wifiAvailable']
X_encoded = X_encoded.drop('wifiAvailable', axis=1)

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, train_size=0.85, test_size=0.15, random_state=1)

# Initialize and train the model
model = XGBRegressor(n_estimators=500, random_state=1)
model.fit(X_train, y_train,
          early_stopping_rounds=5,
          eval_set=[(X_valid, y_valid)],
          verbose=False)

# Evaluate the model
predictions = model.predict(X_valid)
print("Mean absolute error: {:.2f}".format(mean_absolute_error(y_valid, predictions)))

# Prediction analysis
mostLikelyWifi = predictions.max()
leastLikelyWifi = predictions.min()

print("The restaurant most likely to have WiFi has a prediction score of {:.0f}%.".format(mostLikelyWifi*100))
print("The restaurant least likely to have WiFi has a prediction score of {:.0f}%.".format(leastLikelyWifi*100))


'''

Check the result

'''

# Find the businesses corresponding to the highest and lowest predictions
highWifiIndex = np.where(predictions == mostLikelyWifi)[0]
lowWifiIndex = np.where(predictions == leastLikelyWifi)[0]

highWifiRow = X_valid.iloc[highWifiIndex]
lowWifiRow = X_valid.iloc[lowWifiIndex]

print("Business ID most likely to have WiFi:", X.iloc[highWifiRow.index]['business_id'].values)
print("Business ID least likely to have WiFi:", X.iloc[lowWifiRow.index]['business_id'].values)

# Extracting sample reviews for businesses most and least likely to have WiFi
mostLikelyWifiReviews = pd.DataFrame(columns=['text'])
leastLikelyWifiReviews = pd.DataFrame(columns=['text'])

with open("./data/yelp_academic_dataset_review.json", encoding="utf-8") as reviews:
     for _ in range(100000):
        review = reviews.readline()
        if not review:
            break
        reviewData = json.loads(review)
        
        if reviewData['business_id'] in X.iloc[highWifiRow.index]['business_id'].values:
            mostLikelyWifiReviews = mostLikelyWifiReviews._append({'text': reviewData['text']}, ignore_index=True)
        
        if reviewData['business_id'] in X.iloc[lowWifiRow.index]['business_id'].values:
            leastLikelyWifiReviews = leastLikelyWifiReviews._append({'text': reviewData['text']}, ignore_index=True)

print("Most likely to have WiFi, sample reviews:\n", mostLikelyWifiReviews.sample(n=1))
print("Least likely to have WiFi, sample reviews:\n", leastLikelyWifiReviews.sample(n=1))


