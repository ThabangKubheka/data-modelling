import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the training and validation data
bbd_bank_file_path_training = '../Data/Payments Fraud DataSet/transactions_train.csv'
bbd_bank_file_path_validation = '../Data/Payments Fraud DataSet/transactions_test.csv'

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

# Train an Isolation Forest model
bbd_bank_model = IsolationForest(random_state=1)
bbd_bank_model.fit(X_train)

# Predict anomaly scores for validation set
val_anomaly_scores = -bbd_bank_model.decision_function(X_val)

# Prepare the submission format
submission_df = pd.DataFrame({
    'TX_ID': bbd_bank_data_validation['TX_ID'],
    'TX_FRAUD': val_anomaly_scores  # Use the anomaly scores as the fraud scores
})

# Save the submission to a CSV file
submission_file_path = '../Results/ISresult.csv'
submission_df.to_csv(submission_file_path, index=False)

# Visualize the decision tree (optional)
# plt.figure(figsize=(12, 8))
# plot_tree(bbd_bank_model_simple, filled=True, feature_names=X_train.columns, class_names=['Not Fraud', 'Fraud'])
# plt.show()
