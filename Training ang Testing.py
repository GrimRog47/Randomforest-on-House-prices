import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
house_prices = pd.read_csv(r'U:\New folder\Preprocess\DataSets\house price\Cleaned_house_prices_v4.csv')
print("Dataset loaded successfully.")

# split the dataset into features and target variable
X = house_prices.drop('log_amount', axis=1)
y = house_prices['log_amount']
print("Features and target variable separated.")

# convert boolean columns to integers
bool_cols = X.select_dtypes(include=['bool']).columns
for col in bool_cols:
    X[col] = X[col].astype(int)
print("Boolean columns converted to integers.")

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# train the Random Forest Regressor model
print("Training the Random Forest Regressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# predict and convert predictions back to original scale
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)
print("Predictions made and converted back to original scale.")

# evaluate the model
# mean absolute error percentage
mae_percentage = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
print(f"Mean Absolute Error Percentage: {mae_percentage}%")

# feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
import matplotlib.pyplot as plt
feature_importances.plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()

# save the trained model
import joblib
joblib.dump(model, r"U:\New folder\Preprocess\Scripts\House price Project\Model\RandomForest house_price_model_v1.pkl")
print("Trained model saved successfully.")