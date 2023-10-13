import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the training and validation data
bbd_bank_file_path_training = '../input/fraud-modeling/transactions_train.csv'
bbd_bank_file_path_validation = '../input/fraud-modeling/transactions_test.csv'

# Define data types for each column
data_types = {
    'TX_ID': str,
    'TX_TS': str,
    'CUSTOMER_ID': str,
    'TERMINAL_ID': str,
    'TX_AMOUNT': float,
    'TX_FRAUD': int,
    'TRANSACTION_GOODS_AND_SERVICES_AMOUNT': float,
    'TRANSACTION_CASHBACK_AMOUNT': float,
    'CARD_EXPIRY_DATE': str,
    'CARD_DATA': str,
    'CARD_BRAND': str,
    'TRANSACTION_TYPE': str,
    'TRANSACTION_STATUS': str,
    'FAILURE_CODE': str,
    'FAILURE_REASON': str,
    'TRANSACTION_CURRENCY': str,
    'CARD_COUNTRY_CODE': str,
    'MERCHANT_ID': str,
    'IS_RECURRING_TRANSACTION': str,
    'ACQUIRER_ID': str,
}

# Read the training and validation data
bbd_bank_data_training = pd.read_csv(bbd_bank_file_path_training, dtype=data_types)
bbd_bank_data_validation = pd.read_csv(bbd_bank_file_path_validation, dtype=data_types)

# Define features and target variable
bbd_bank_features = ['TX_AMOUNT', 'TRANSACTION_GOODS_AND_SERVICES_AMOUNT', 'TRANSACTION_CASHBACK_AMOUNT',
                     'CARD_BRAND', 'TRANSACTION_TYPE', 'TRANSACTION_STATUS', 'FAILURE_CODE', 'FAILURE_REASON',
                     'TRANSACTION_CURRENCY', 'CARD_COUNTRY_CODE', 'IS_RECURRING_TRANSACTION', 'CARDHOLDER_AUTH_METHOD']

X_train = pd.get_dummies(bbd_bank_data_training[bbd_bank_features])
y_train = bbd_bank_data_training['TX_FRAUD']

X_val = pd.get_dummies(bbd_bank_data_validation[bbd_bank_features])

# Align the columns in the training and validation sets
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)

# Random Forest Regressor model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
fraud_preds = forest_model.predict(X_val)

submission_df = pd.DataFrame({
    'TX_ID': bbd_bank_data_validation['TX_ID'],
    'TX_FRAUD': fraud_preds  # Use the predicted probabilities as the fraud scores
})

# Save the submission to a CSV file
submission_file_path = 'zubmission.csv'
submission_df.to_csv(submission_file_path, index=False)
