# Objective
# To predict whether customer will churn or not
# ------------------------------------------------

# ~~~~~ IMPORTING LIBRARIES ~~~~~~~~~~~~~~~~~~~~~

import keras.callbacks
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from helpers.custom_metrics import f1_m
import seaborn as sns
import collections
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Import df
# ------------------------------------------------
df = pd.read_csv("./data/AV/train_PDjVQMB.csv")
X_test = pd.read_csv("./data/AV/test_lTY72QC.csv")
X_test_copy = X_test.copy()

# Dependent(y) and Independent(X) variables
# ------------------------------------------------
y = df['Is_Churn']
X = df.drop('Is_Churn', axis=1)


# ~~~~~ EXPLORATORY DATA ANALYSIS ~~~~~~~~~~~~~~~~~~~~~

# Pie Plot for Target Variable
# ------------------------------------------------
counts = df['Is_Churn'].value_counts()
labels = 'No_Churn', 'Churn'
data = [counts[0], counts[1]]
plt.pie(data, labels=labels, autopct='%.0f%%', shadow=True)
plt.title("Bank Customers Churned and Retained", size=15)
plt.show()

# Categorical variables Relationships w.r.t target variable
# ---------------------------------------------------------------
# Initial Insights
# Vintage with bank more, customers less churn
# Customers with many product holdings (3+) churn less
# Poor credit rating customers churn more

fig, axarr = plt.subplots(2, 3, figsize=(20, 12))
sns.countplot(x='Gender', hue='Is_Churn', data=df, ax=axarr[0][0])
sns.countplot(x='Income', hue='Is_Churn', data=df, ax=axarr[0][1])
sns.countplot(x='Vintage', hue='Is_Churn', data=df, ax=axarr[0][2])
sns.countplot(x='Transaction_Status', hue='Is_Churn', data=df, ax=axarr[1][0])
sns.countplot(x='Product_Holdings', hue='Is_Churn', data=df, ax=axarr[1][1])
sns.countplot(x='Credit_Category', hue='Is_Churn', data=df, ax=axarr[1][2])
plt.show()

# EDA - Numerical variables Relationships w.r.t target variable
# ---------------------------------------------------------------
sns.boxplot(y='Balance', x='Is_Churn', hue='Is_Churn', data=df)
plt.show()
sns.boxplot(y='Age', x='Is_Churn', hue='Is_Churn', data=df)
plt.show()


# ~~~~~ DATA PREPROCESSING ~~~~~~~~~~~~~~~~~~~~~~~~~~

# Check for missing data
# ------------------------------------------------
print(X.isnull().sum())
print(X_test.isnull().sum())
print(y.isnull().sum())

# Drop feature not relevant for prediction
# -------------------------------------------------------
X = X.drop('ID', axis=1)
X_test = X_test.drop('ID', axis=1)
print("X shape", X.shape)
print("X test shape", X_test.shape)

# Label encoding - Ordinal variables
# ------------------------------------------
cols = ['Gender', 'Income', 'Product_Holdings', 'Credit_Category']

def label_encode_func(input):
    input['Gender'].replace({'Female': 0, 'Male': 1}, inplace=True)
    input['Income'].replace({'Less than 5L': 0, '5L - 10L': 1, '10L - 15L': 2, 'More than 15L': 3}, inplace=True)
    input['Product_Holdings'].replace({'1': 0, '2': 1, '3+': 2}, inplace=True)
    input['Credit_Category'].replace({'Poor': 0, 'Average': 1, 'Good': 2}, inplace=True)

# Encoding for train df
label_encode_func(X)
# Encoding for test df
label_encode_func(X_test)

# Imbalanced classes, use SMOTE to oversample minority class
# ----------------------------------------------------------
counter = collections.Counter(y)

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# Feature Scaling
# ---------------------------------------------
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


# ~~~~~ MODELLING (Import, Instantiate, Fit, Predict)~~~~~~~~~~~~~~~~~~~~~

# Define ANN Architecture
# --------------------------------------------------
model_ann = tf.keras.models.Sequential()

model_ann.add(tf.keras.layers.Dense(units=10, activation='relu', ))
model_ann.add(tf.keras.layers.Dense(units=5, activation='relu', ))

model_ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile & Train the model
# ----------------------------------------------
model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_m])

history = model_ann.fit(X, y, epochs=100, validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                                                       patience=10,
                                                       mode="min")])
model_ann.summary()


# ~~~~~ PREDICTIONS ~~~~~~~~~~~~~~~~~~~~~

# Prediction on test set
# ---------------------------
yp = model_ann.predict(X_test)
y_pred = []
for rzlt in yp:
    if rzlt > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# Change to format for submission
# ------------------------------------
y_pred_df = pd.DataFrame(y_pred, columns={'Is_Churn'})
rzlt_df = y_pred_df.join(X_test_copy['ID'])
new_cols = ["ID", "Is_Churn"]
rzlt_df = rzlt_df[new_cols]

# Export to csv
# ----------------------------
rzlt_df.to_csv("C:\\Users\\sylvi\\OneDrive\\Desktop\\m2.csv", index=False, encoding='utf-8')
