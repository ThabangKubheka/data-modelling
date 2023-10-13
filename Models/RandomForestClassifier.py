import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve

# Step 1: Data Preprocessing
# Load the datasets
bbd_bank_file_path_training = '../Data/Payments Fraud DataSet/transactions_train.csv'
bbd_bank_file_path_validation = '../Data/Payments Fraud DataSet/transactions_test.csv'
bbd_bank_file_path_merchants = '../Data/Payments Fraud DataSet/merchants.csv'
bbd_bank_file_path_customers = '../Data/Payments Fraud DataSet/customers.csv'
bbd_bank_file_path_terminals = '../Data/Payments Fraud DataSet/terminals.csv'

# Setting Datatypes for the data

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
    'IS_RECURRING_TRANSACTION': object,
    'ACQUIRER_ID': str,
}

customers = pd.read_csv(bbd_bank_file_path_customers)
terminals = pd.read_csv(bbd_bank_file_path_terminals)
merchants = pd.read_csv(bbd_bank_file_path_merchants)
transactions_train = pd.read_csv(bbd_bank_file_path_training, dtype=data_types)
transactions_test = pd.read_csv(bbd_bank_file_path_validation, dtype=data_types)

# Addressing datatype issues
customers['CUSTOMER_ID'] = customers['CUSTOMER_ID'].astype(str)
terminals['TERMINAL_ID'] = terminals['TERMINAL_ID'].astype(str)

# Merge datasets based on common identifiers
# Assuming CUSTOMER_ID, TERMINAL_ID, and MERCHANT_ID are relevant for merging
train_data = pd.merge(transactions_train, customers, on='CUSTOMER_ID', how='left')
train_data = pd.merge(train_data, terminals, on='TERMINAL_ID', how='left')
train_data = pd.merge(train_data, merchants, on='MERCHANT_ID', how='left')

test_data = pd.merge(transactions_test, customers, on='CUSTOMER_ID', how='left')
test_data = pd.merge(test_data, terminals, on='TERMINAL_ID', how='left')
test_data = pd.merge(test_data, merchants, on='MERCHANT_ID', how='left')

# Dropping columns we don't need
test_data.drop('ID_JOIN', axis=1, inplace=True)


# Step 2: Feature Engineering
# Calculate Euclidean distance for each transaction
def calculate_euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Calculate Euclidean distance for each transaction
train_data['EUCLIDEAN_DISTANCE'] = calculate_euclidean_distance(train_data['x_customer_id'],
                                                                train_data['y_customer_id'],
                                                                train_data['x_terminal_id'],
                                                                train_data['y_terminal__id'])

test_data['EUCLIDEAN_DISTANCE'] = calculate_euclidean_distance(test_data['x_customer_id'],
                                                               test_data['y_customer_id'],
                                                               test_data['x_terminal_id'],
                                                               test_data['y_terminal__id'])

# Set a threshold for anomaly detection
euclidean_distance_threshold = 10.0

# Flag potential anomalies based on the threshold
train_data['POTENTIAL_ANOMALY'] = train_data['EUCLIDEAN_DISTANCE']

# Decide to remove non-numerical features as these are breaking the model
features = [
    'TX_AMOUNT',
    'TRANSACTION_GOODS_AND_SERVICES_AMOUNT',
    'TRANSACTION_CASHBACK_AMOUNT',
    'x_customer_id',
    'y_customer_id',
    'x_terminal_id',
    'y_terminal__id',
    'EUCLIDEAN_DISTANCE',
]

# Step 3: Model Selection and Training
X_train, X_validation, y_train, y_validation = train_test_split(
    train_data[features], train_data['TX_FRAUD'], test_size=0.2, random_state=42,stratify=train_data['TX_FRAUD'])

# Impute missing values using mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_validation_imputed = imputer.transform(X_validation)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_imputed, y_train)

# Step 4: Model Evaluation
# Evaluate the model's performance using validation data
validation_predictions = model.predict_proba(X_validation_imputed)[:, 1]
validation_auc = roc_auc_score(y_validation, validation_predictions)
print('AUC on validation data:', validation_auc)

# Calculate binary predictions based on a threshold of 0.5
validation_predictions_binary = (validation_predictions >= 0.5).astype(int)

# Calculate validation accuracy, precision, and recall
validation_accuracy = accuracy_score(y_validation, validation_predictions_binary)
validation_precision = precision_score(y_validation, validation_predictions_binary)
validation_recall = recall_score(y_validation, validation_predictions_binary)

print('Validation Accuracy:', validation_accuracy)
print('Validation Precision:', validation_precision)
print('Validation Recall:', validation_recall)

# Step 5: Predict on Test Data
# Evaluate the model's performance using validation data
test_data_imputed = imputer.transform(test_data[features])
test_predictions = model.predict_proba(test_data_imputed)[:, 1]

# Preparing for visualization
# Create an array of transaction indices
transaction_indices = np.arange(len(test_data))

# Separate the test data into fraud and non-fraud based on predicted probabilities
fraud_indices = transaction_indices[test_predictions >= 0.5]
non_fraud_indices = transaction_indices[test_predictions < 0.5]

# Step 6: Prepare Submission File
submission_df = test_data[['TX_ID']].copy()
submission_df['TX_FRAUD'] = test_predictions
submission_df.to_csv('../Results/RFC.csv', index=False)

# Step 7 : Plot a scatter plot for the result
plt.figure(figsize=(10, 6))
plt.scatter(fraud_indices, test_predictions[fraud_indices], color='red', label='Fraud')
plt.scatter(non_fraud_indices, test_predictions[non_fraud_indices], color='green', label='Non-Fraud')

plt.xlabel('Transaction Index')
plt.ylabel('Predicted Probability')
plt.title('Predicted Probabilities for Test Data')
plt.legend()
plt.savefig('../Plots/scatter_plot.png')
plt.show()

# Step 8 : Plot a ROC Curve
fpr, tpr, _ = roc_curve(y_validation, validation_predictions)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('../Plots/roc_curve.png')
plt.show()

# Step 9 : Feature Importance Plot
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
features_sorted = np.array(features)[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_sorted)
plt.xlabel('Feature Importance')
plt.title('Feature Importance')
plt.savefig('../Plots/feature_importance_plot.png')
plt.show()

