# train.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib  # for saving the model
import pandas as pd

# Load the preprocessed data
X = pd.read_csv('./data/train.csv')
X_encoded = pd.get_dummies(X.drop(['business_id'], axis=1), drop_first=True)

# Separate features and target variable
y = X_encoded.pop('highProbAttr')

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, train_size=0.85, test_size=0.15, random_state=1)

# Initialize and train the model
model = XGBRegressor(n_estimators=500, random_state=1)
model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)

# Save the trained model
joblib.dump(model, './models/attr_model.xgb')

# Evaluate the model
predictions = model.predict(X_valid)
mae = mean_absolute_error(y_valid, predictions)
print(f"Mean absolute error: {mae:.2f}")

# Identify predictions with extreme attribute availability likelihoods
mostLikelyAttr = predictions.max()
leastLikelyAttr = predictions.min()
print(f"The business most likely to have the attribute has a prediction score of {mostLikelyAttr * 100:.0f}%.")
print(f"The business least likely to have the attribute has a prediction score of {leastLikelyAttr * 100:.0f}%.")