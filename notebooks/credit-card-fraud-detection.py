# Import  the module

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

# Loading the dataset

df = pd.read_csv("/content/drive/MyDrive/creditcard.csv")
df.head()

# Checking EDA
# statistical info
df.describe()

# Datatype of Dataset information
df.info()

# Check the null values of dataset
df.isnull().sum()

# Exploratory Data Analysis of data

sns.countplot(df['Class'])

df_temp = df.drop(columns=['Time', 'Amount', 'Class'], axis=1)

# create dist plots
fig, ax = plt.subplots(ncols=4, nrows=7, figsize=(20, 50))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df_temp[col], ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5)

sns.distplot(df['Time'])

sns.distplot(df['Amount'])

#  Checking Coorelation Matrix

corr = df.corr()
plt.figure(figsize=(30,40))
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Split the Input

X = df.drop(columns=['Class'], axis=1)
y = df['Class']

# Scaling
# Standard Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaler = sc.fit_transform(X)

x_scaler[-1]

# Training the  Model

# Spliting the data into trainig and testing
# train test split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.25, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train the model with training Data
model.fit(x_train, y_train)

# Test the model with Testing Data
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))

# Importing  RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# Training the model
model.fit(x_train, y_train)

# Testing the created model
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))

# Importing XGBClassifier
from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)

# training the model
model.fit(x_train, y_train)

# testing the model
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))
print("simpi",y_test,y_pred)

# Class Imbalancement

sns.countplot(y_train)

# hint - use combination of over sampling and under sampling
# balance the class with equal distribution
from imblearn.over_sampling import SMOTE
over_sample = SMOTE()
x_smote, y_smote = over_sample.fit_resample(x_train, y_train)

sns.countplot(y_smote)

#Import LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# training
model.fit(x_smote, y_smote)

# testing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)

# training the model
model.fit(x_smote, y_smote)

# testing the model
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))

from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)

# training the model
model.fit(x_smote, y_smote)

# testing the model
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))

#Deployment

!pip install streamlit --quiet

Commented out IPython magic to ensure Python compatibility.
%%writefile app.py
import streamlit as st
import pickle

st.title("Credit Card Fraud Detection")

# Add a single input field for a binary choice (0 or 1)
user_input = st.text_input("Enter 0 for Legitimate, 1 for Fraud", "")

# Check if the input is a valid digit (0 or 1)
if user_input.isdigit() and user_input in ['0', '1']:
    # Convert the input to an integer
    user_input = int(user_input)

    # Add submit button
    if st.button("Submit"):
        # Display result based on the input value
        if user_input == 0:
            result = "Legitimate"
        else:
            result = "Fraud"

        st.write("Result:", result)

else:
    st.warning("Please enter a valid digit (0 or 1).")


!streamlit run app.py & npx localtunnel --port 8501

