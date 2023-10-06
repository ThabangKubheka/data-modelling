import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt

# Getting the data and reading it
bbd_bank_file_path = 'Data/train.csv'
bbd_bank_data = pd.read_csv(bbd_bank_file_path)

# Assumptions:
# target is the target variable indicating fraud (1) or not (0)


print("-------------------------------------------------------------------------")
print("-----------Model1 :Decision Tree Regressor-------------------------------")
print("-------------------------------------------------------------------------")

# Model1 :Decision Tree Regressor:
# Creating the model
# Step 1:  Define model
bbd_bank_model = DecisionTreeRegressor(random_state=1)

# Prediction Target
y = bbd_bank_data.SalePrice

# Train/Fit model
# Define features
bbd_bank_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Define training dataset
X = bbd_bank_data[bbd_bank_features]

# fit model
bbd_bank_model.fit(X, y)

# Predict
predictions = bbd_bank_model.predict(X.head())
print("Making predictions for the following 5 transactions:")
print(X.head())
print("The predictions are")
print(predictions)

# Validation
predicted_sale_prices = bbd_bank_model.predict(X)
print("The predicted sale prices  are:")
print(predicted_sale_prices)

# Mean absolute error
mae = mean_absolute_error(y, predicted_sale_prices)
print("The mean absolute error:", mae)

# Specify Model Again
bbd_bank_model = DecisionTreeRegressor(random_state=1)

# Splitting data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Fit model again
bbd_bank_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = bbd_bank_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
bbd_bank_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
bbd_bank_model.fit(train_X, train_y)
val_predictions = bbd_bank_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

print("-------------------------------------------------------------------------")
print("-------------------------------End---------------------------------------")
print("-------------------------------------------------------------------------")
print()

print("-----------Model-2:Random Forest Regressor:-------------------------------")
print("-------------------------------------------------------------------------")
# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_predictions)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

print("-------------------------------------------------------------------------")
print("-------------------------------End---------------------------------------")

print()
print("-------------------------------------------------------------------------")
print("-----------MModel-3: XGBoost Regressor-------------------------------")
print("-------------------------------------------------------------------------")
# Define the model and set parameters
xgb_model = xgb.XGBRegressor(random_state=1)

# Fit the XGBoost model
xgb_model.fit(train_X, train_y)

# Predictions and calculate mean absolute error for XGBoost model on validation data
xgb_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(val_y, xgb_predictions)

print("Validation MAE for XGBoost Model: {}".format(xgb_val_mae))

print("-------------------------------------------------------------------------")
print("-------------------------------End---------------------------------------")
print("-------------------------------------------------------------------------")

# Results:
# Let's review the mean absolute errors (MAE) for each model obtained on the validation data:
#
# Decision Tree Regressor:
# Validation MAE: Varies based on max_leaf_nodes. The best validation MAE achieved was approximately 24252 when
# max_leaf_nodes=100.

# Random Forest Regressor:
# Validation MAE: Approximately 17713.6.

# XGBoost Regressor:
# Validation MAE: Approximately 23776.1.

# In this case, the Random Forest Regressor achieved the lowest validation
# MAE, making it the best-performing model among the three. A lower MAE indicates that the Random Forest model's
# predictions were, on average, closer to the actual sale prices on the validation data compared to the other models.


# # Extract feature importances from the Random Forest model
feature_importances_rf = rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importances_df_rf = pd.DataFrame({
    'Feature': bbd_bank_features,
    'Importance': feature_importances_rf
})

# Sort features by importance (descending order)
feature_importances_df_rf = feature_importances_df_rf.sort_values(by='Importance', ascending=False)

print("Random Forest Model - Feature Importances:")
print(feature_importances_df_rf)

# # Extract feature importances from the XGBoost model
feature_importances_xgb = xgb_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importances_df_xgb = pd.DataFrame({
    'Feature': bbd_bank_features,
    'Importance': feature_importances_xgb
})

# Sort features by importance (descending order)
feature_importances_df_xgb = feature_importances_df_xgb.sort_values(by='Importance', ascending=False)

print("XGBoost Model - Feature Importances:")
print(feature_importances_df_xgb)


# Plot feature importances for Random Forest model
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df_rf['Feature'], feature_importances_df_rf['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Random Forest Model - Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# Plot feature importances for XGBoost model
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_df_xgb['Feature'], feature_importances_df_xgb['Importance'], color='lightgreen')
plt.xlabel('Importance')
plt.title('XGBoost Model - Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# Create a summary DataFrame for MAE comparison
mae_summary = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'XGBoost'],
    'Validation MAE': [val_mae, rf_val_mae, xgb_val_mae]
})

# Plot a bar chart to show validation MAE comparison
plt.figure(figsize=(8, 6))
plt.bar(mae_summary['Model'], mae_summary['Validation MAE'], color=['lightblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Model')
plt.ylabel('Validation MAE')
plt.title('Validation MAE Comparison')
plt.show()

# Display the summary DataFrame
print("Validation MAE Summary:")
print(mae_summary)

