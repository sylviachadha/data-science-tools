# Objective
# To predict whether customer will churn or not
#------------------------------------------------

# ~~~~~ DATA PREPROCESSING ~~~~~~~~~~~~~~~~~~~~~

# Import Libraries
#--------------------------------------------
import keras.callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import Dataset
#------------------------------------------------
dataset = pd.read_csv("./data/AV/train_PDjVQMB.csv")
print("dataset shape", dataset.shape)
dataset.head(5)
X_df_copy = dataset.copy()
# Upload Test Dataset
X_test = pd.read_csv("./data/AV/test_lTY72QC.csv")
X_test_df_copy = X_test.copy()

# Dependent(y) and Independent(X) variables
#------------------------------------------------
y = dataset['Is_Churn']
X = dataset.drop('Is_Churn', axis=1)
counts = dataset['Is_Churn'].value_counts()

labels = 'No_Churn','Churn'
data = [counts[0],counts[1]]
plt.pie(data, labels = labels,autopct='%.0f%%',shadow=True)
plt.title("Bank Customers Churned and Retained", size=15)
plt.show()

# EDA - Categorical variables Relationships w.r.t target variable
#---------------------------------------------------------------
# Observations
# As from plots, vintage with bank more, customers less churn
# Customers with many product holdings (3+) churn less
# Poor credit rating customers churn more

import seaborn as sns
fig, axarr = plt.subplots(2,3,figsize=(20,12))
sns.countplot(x='Gender', hue='Is_Churn', data=dataset, ax=axarr[0][0])
sns.countplot(x='Income', hue='Is_Churn', data=dataset, ax=axarr[0][1])
sns.countplot(x='Vintage', hue='Is_Churn', data=dataset, ax=axarr[0][2])
sns.countplot(x='Transaction_Status', hue='Is_Churn', data=dataset, ax=axarr[1][0])
sns.countplot(x='Product_Holdings', hue='Is_Churn', data=dataset, ax=axarr[1][1])
sns.countplot(x='Credit_Category', hue='Is_Churn', data=dataset, ax=axarr[1][2])
plt.show()


# EDA - Numerical variables Relationships w.r.t target variable
#---------------------------------------------------------------

sns.boxplot(y='Balance', x='Is_Churn', hue='Is_Churn', data=dataset)
plt.show()
sns.boxplot(y='Age', x='Is_Churn', hue='Is_Churn', data=dataset)
plt.show()

# Check for missing data
#------------------------------------------------
print(X.isnull().sum())
print(X_test.isnull().sum())
print(y.isnull().sum())

# Drop feature/column not relevant for prediction
#-------------------------------------------------------
X = X.drop('ID',axis=1)
X_test = X_test.drop('ID',axis=1)
print("X shape",X.shape)
print("X test shape",X_test.shape)

# Check unique values in categorical columns & label encode
#-----------------------------------------------------------------
X.dtypes
cols = ['Gender', 'Income', 'Product_Holdings', 'Credit_Category']
for columns in cols:
    print(X[columns].unique())

X['Gender'].replace({'Female': 0, 'Male': 1},inplace=True)
X['Income'].replace({'Less than 5L': 0,'5L - 10L': 1, '10L - 15L': 2, 'More than 15L': 3 },inplace=True)
X['Product_Holdings'].replace({'1': 0, '2': 1, '3+': 2},inplace=True)
X['Credit_Category'].replace({'Poor': 0, 'Average': 1, 'Good':2},inplace=True)


# Same Label encoding required for test data
X_test.dtypes
cols = ['Gender', 'Income', 'Product_Holdings', 'Credit_Category']
for columns in cols:
    print(X_test[columns].unique())

X_test['Gender'].replace({'Female': 0, 'Male': 1},inplace=True)
X_test['Income'].replace({'Less than 5L': 0,'5L - 10L': 1, '10L - 15L': 2, 'More than 15L': 3 },inplace=True)
X_test['Product_Holdings'].replace({'1': 0, '2': 1, '3+': 2},inplace=True)
X_test['Credit_Category'].replace({'Poor': 0, 'Average': 1, 'Good':2},inplace=True)


# Feature Scaling
#-----------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# ~~~~~ MODELLING (Import, Instantiate, Fit, Predict)~~~~~~~~~~~~~~~~~~~~~

import warnings
warnings.filterwarnings("ignore")

# Define ANN Architecture
#--------------------------------------------------
model_ann = tf.keras.models.Sequential()

model_ann.add(tf.keras.layers.Dense(units=10, activation='relu',))
model_ann.add(tf.keras.layers.Dense(units=5, activation='relu',))

model_ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Compile & Train the model
#----------------------------------------------
# Compile
model_ann.compile(optimizer = 'adam',loss= 'binary_crossentropy',
            metrics=['accuracy'])


# Fit
model_ann.fit(X, y, epochs=100,validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                                                       patience=10,
                                                       mode="min")])
model_ann.summary()

# Prediction on test set
#---------------------------
yp = model_ann.predict(X_test)
y_pred=[]
for rzlt in yp:
    if rzlt>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# We need a dataframe with ID and result prediction of churn or no_churn
y_pred_df = pd.DataFrame(y_pred,columns={'Is_Churn'})
rzlt_df = y_pred_df.join(X_test_df_copy['ID'])
new_cols = ["ID","Is_Churn"]
rzlt_df = rzlt_df[new_cols]

rzlt_values = (rzlt_df['Is_Churn'].value_counts())
print(rzlt_values)
print("% Churn",rzlt_values[1]/ rzlt_values[0]*100)

# Export to excel
rzlt_df.to_csv("C:\\Users\\sylvi\\OneDrive\\Desktop\\m1.csv")


# ~~~~~ PERFORMANCE METRIC (Macro F1 score)~~~~~~~~~~~~~


