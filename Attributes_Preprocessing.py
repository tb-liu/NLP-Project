# preprocessing.py
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI



# Initialize the LangChain with OpenAI API key
OPENAI_API_KEY = "sk-proj-N662AbswIp5Bihu7DsVkT3BlbkFJuSprqYzZkzNNMzjN3TCV"
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

# Example attributes for keyword generation
attributes = "kid-friendly"

# Define the prompt for generating lowercase keywords
keywords_prompt = PromptTemplate.from_template(
    "Generate 15 lowercase keywords related to these attributes: {attributes}. List them as comma-separated values."
)
keywords_chain = LLMChain(llm=llm, prompt=keywords_prompt, output_key='generated_keywords')

# Define the prompt for suggesting a star rating threshold
rating_prompt = PromptTemplate.from_template(
    "You are a sentiment analysis machine, here is a word:{attributes}, return >3 if the comment with this word is neutral or good, return <3 if the comment with this word is negative. You can only respond with: <3 or >3"
)
rating_chain = LLMChain(llm=llm, prompt=rating_prompt, output_key='rating_threshold')

# Combine the two chains into a SequentialChain
attribute_analysis_chain = SequentialChain(
    chains=[keywords_chain, rating_chain],
    input_variables=['attributes'],
    output_variables=['generated_keywords', 'rating_threshold'],
    verbose=True
)

result = attribute_analysis_chain(attributes)

# Extract keywords and rating threshold from result
words = result['generated_keywords'].split(', ')
rating_threshold = result['rating_threshold'].strip()

print(words)
print(rating_threshold)



# Parse the rating threshold
comparison_operator = rating_threshold[0]
star_threshold = int(rating_threshold[1:])

# Load review data
highProbAttrReviewsData = []
normalReviewsData = []


with open("./data/yelp_academic_dataset_review.json", encoding="utf-8") as reviews:
    for review in reviews:
        reviewData = json.loads(review)
        
        # Check review text for keywords and apply the parsed star rating threshold
        if any(word in reviewData['text'].lower() for word in words) and eval(f"reviewData['stars'] {comparison_operator} star_threshold"):
            highProbAttrReviewsData.append({'business_id': reviewData['business_id'], 'date': reviewData['date'], 'text': reviewData['text']})
        
        # Randomly pick approximately one in every 180 reviews without key words as normal reviews
        if np.random.randint(1, 180) == 1 and not any(word in reviewData['text'].lower() for word in words):
            normalReviewsData.append({'business_id': reviewData['business_id'], 'date': reviewData['date'], 'text': reviewData['text']})

highProbAttrReviews = pd.DataFrame(highProbAttrReviewsData, columns=['business_id', 'date', 'text'])
normalReviews = pd.DataFrame(normalReviewsData, columns=['business_id', 'date', 'text'])

# Function to format data from business IDs
def formatDataFromID(business_ids, highProbAttr=0):
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
                    businessData['highProbAttr'] = highProbAttr
                    businessDataList.append(businessData)
    
    return pd.DataFrame(businessDataList)

highProbAttrBusinesses = highProbAttrReviews['business_id'].tolist()
normalBusinesses = normalReviews['business_id'].tolist()

X_highProbAttr = formatDataFromID(highProbAttrBusinesses, 1)
X_normal = formatDataFromID(normalBusinesses)

# Merge data related to attribute availability and non-availability
X = pd.concat([X_highProbAttr, X_normal], ignore_index=True)

# Dropping columns not relevant to attribute prediction
X_clean = X.drop(['categories', 'category_Restaurants'], axis=1)

# Handling missing data and encoding categorical variables
cols_with_missing = [col for col in X.columns if X[col].isnull().sum() > X.shape[0] * 0.9]
X_clean = X_clean.drop(cols_with_missing, axis=1).fillna(0)

# Save the preprocessed data
X_clean.to_csv('./data/preprocessed_attr_data.csv', index=False)

# divide into train and test

# Load the data from CSV
data = pd.read_csv('./data/preprocessed_attr_data.csv')

# Split the data into a 70% training set and a 30% test set
train, test = train_test_split(data, test_size=0.3, random_state=42)  # Use a random state for reproducibility

# Save the training and test sets to CSV files
train.to_csv('./data/train.csv', index=False)
test.to_csv('./data/test.csv', index=False)

