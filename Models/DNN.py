import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score

plt.rcParams["figure.dpi"] = 200


def build_transactions(transactions_frame, customers_frame, terminals_frame, merchants_frame):
    transactions_frame = transactions_frame.join(customers_frame.set_index("CUSTOMER_ID"), "CUSTOMER_ID", 'left',
                                                 lsuffix='_caller', rsuffix='_other')
    transactions_frame = transactions_frame.join(terminals_frame.set_index("TERMINAL_ID"), "TERMINAL_ID", 'left',
                                                 lsuffix='_caller', rsuffix='_other')
    transactions_frame = transactions_frame.join(merchants_frame.set_index("MERCHANT_ID"), "MERCHANT_ID", 'left',
                                                 lsuffix='_caller', rsuffix='_other')
    transactions_frame = transactions_frame.drop(columns=["CUSTOMER_ID", "TERMINAL_ID", "MERCHANT_ID"]).fillna(0)
    return transactions_frame


def get_field_uniqueness(data_frame):
    detail_list = []
    for col in data_frame.columns:
        detail_list.append([col, data_frame[col].nunique()])

    return detail_list


def generate_one_hot(frame):
    detail_list = get_field_uniqueness(frame)
    encodeable = list(filter(lambda x: x[1] < 20, detail_list))
    for item in encodeable:
        print(item)
        if item[1] > 2:
            frame = pd.get_dummies(frame, columns=[item[0], ])
        else:
            frame = pd.get_dummies(frame, columns=[item[0]], drop_first=True)
    return frame

customers = pd.read_csv('./data/customers.csv')
terminals = pd.read_csv('./data/terminals.csv')
merchants = pd.read_csv('./data/merchants.csv')
transactions = pd.read_csv('./data/transactions_train.csv')

transactions["TX_FRAUD"] = transactions["TX_FRAUD"].astype("float32")

target = pd.DataFrame(transactions['TX_FRAUD']).astype('float32')
result_set = target.to_numpy()

transactions = transactions[["CUSTOMER_ID", "TERMINAL_ID", "MERCHANT_ID", "TX_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT", "TRANSACTION_STATUS", "TRANSACTION_TYPE", "TRANSACTION_CURRENCY"]]
transactions = generate_one_hot(transactions)

print()

merchants = merchants[["MERCHANT_ID", "BUSINESS_TYPE", "OUTLET_TYPE", "DEPOSIT_PERCENTAGE", "TAX_EXCEMPT_INDICATOR"]]
merchants = generate_one_hot(merchants)

print()

transactions = build_transactions(transactions, customers, terminals, merchants)

transactions = transactions.sort_index()
train_set = transactions.astype('float32').to_numpy()


# split validation and training
X_train, X_val, y_train, y_val = train_test_split(train_set, result_set.ravel(), test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(36, 16, 2), random_state=42)

clf.fit(X_train, y_train)

#Scoring the data
print("training", clf.score(X_train, y_train))
print("validation", clf.score(X_val, y_val))

test_data = pd.read_csv('./data/transactions_test.csv')

test_result = test_data[["TX_ID"]]
test_result["TX_FRAUD"] = 0

test_data = test_data[["CUSTOMER_ID", "TERMINAL_ID", "MERCHANT_ID", "TX_AMOUNT", "TRANSACTION_CASHBACK_AMOUNT", "TRANSACTION_STATUS", "TRANSACTION_TYPE", "TRANSACTION_CURRENCY"]]
test_data = generate_one_hot(test_data)
test_data = build_transactions(test_data, customers, terminals, merchants)

missing_columns = set( transactions.columns ) - set( test_data.columns )

for column in missing_columns :
  test_data["missing_columns"] = 0

  test_array = test_data.astype('float32').to_numpy()

  # apply same transformation to test data
  test_array = scaler.transform(test_array)

  test_result["TX_FRAUD"] = clf.predict(test_array)
  test_result.to_csv("./submission.csv", index=False)

  fraud_list_x = []
  fraud_list_y = []
  normal_list_x = []
  normal_list_y = []

  fraud_map = {}

  for index in transactions.index:
      cust_x = np.round(transactions.loc[index, "x_customer_id"])
      cust_y = np.round(transactions.loc[index, "y_customer_id"])

      try:
          fraud_map[cust_x]
          try:
              fraud_map[cust_x][cust_y]
          except:
              fraud_map[cust_x][cust_y] = 0
      except:
          fraud_map[cust_x] = {cust_y: 0}

      if target.loc[index, "TX_FRAUD"] == 1:
          fraud_list_x.append(cust_x)
          fraud_list_y.append(cust_y)
          fraud_map[cust_x][cust_y] += 1
      else:
          normal_list_x.append(cust_x)
          normal_list_y.append(cust_y)

  plt.scatter(normal_list_x, normal_list_y, c='blue', s=1, alpha=0.1)
  plt.show()

  plt.scatter(fraud_list_x, fraud_list_y, c='red', s=1, alpha=0.1)
  plt.show()

  plt.scatter(normal_list_x, normal_list_y, c='blue', s=1, alpha=0.1)
  plt.scatter(fraud_list_x, fraud_list_y, c='red', s=1, alpha=0.1)
  plt.show()


from numpy import linspace, meshgrid
from scipy.interpolate import griddata
import plotly.graph_objects as go

# Make data.
X = []
Y = []
Z = []

for _X in fraud_map.keys() :
  for _Y in fraud_map[_X].keys() :
    X.append(_X)
    Y.append(_Y)
    Z.append(fraud_map[_X][_Y])

X = np.array(X)
Y = np.array(Y)

##X, Y = np.meshgrid(X, Y)
print(len(Z))
Z = np.array(Z)

fig = go.Figure()
fig.add_trace(go.Contour(x=X,y=Y,z=Z,line_smoothing=0))
fig.update_layout(autosize=False)
##fig.add_trace(go.Scatter(x=y,=y=y,text=z,mode='markers+text'))
#

fraud_list_x = []
fraud_list_y = []
normal_list_x = []
normal_list_y = []

for index in transactions.index:
    term_x = transactions.loc[index, "x_terminal_id"]
    term_y = transactions.loc[index, "y_terminal__id"]

    if target.loc[index, "TX_FRAUD"] == 1:
        fraud_list_x.append(term_x)
        fraud_list_y.append(term_y)
    else:
        normal_list_x.append(term_x)
        normal_list_y.append(term_y)

plt.scatter(normal_list_x, normal_list_y, c='blue', s=1, alpha=0.5)
plt.show()

plt.scatter(fraud_list_x, fraud_list_y, c='red', s=1, alpha=0.5)
plt.show()

plt.scatter(normal_list_x, normal_list_y, c='blue', s=1, alpha=0.5)
plt.scatter(fraud_list_x, fraud_list_y, c='red', s=1, alpha=0.5)
plt.show()




transactions = pd.read_csv('./data/transactions_train.csv')

for index in transactions.index:
  amount = transactions.loc[index, "TX_AMOUNT"]
  currency = transactions.loc[index, "TRANSACTION_CURRENCY"]

  if target.loc[index, "TX_FRAUD"] == 1 :
    fraud_list_x.append(term_x)
    fraud_list_y.append(term_y)
  else :
    normal_list_x.append(term_x)
    normal_list_y.append(term_y)

plt.scatter(normal_list_x, normal_list_y, c='blue', s=1, alpha=0.5)
plt.show()

plt.scatter(fraud_list_x, fraud_list_y, c='red', s= 1, alpha=0.5)
plt.show()

plt.scatter(normal_list_x, normal_list_y, c='blue', s=1, alpha=0.5)
plt.scatter(fraud_list_x, fraud_list_y, c='red', s= 1, alpha=0.5)
plt.show()