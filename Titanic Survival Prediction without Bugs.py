# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 04:51:46 2025

@author: hongf
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Intentional mistake: wrong URL for Titanic dataset
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_data():
    # Load the Titanic dataset
    df = pd.read_csv(DATA_URL)

    # Introduce NaNs intentionally
    df.loc[5:10, 'Age'] = np.nan
    return df

def preprocess(df):
    
    y = df['Survived']
    # Drop irrelevant columns
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Survived'])  # OK
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])  # NaNs still exist here
    df['Fare'] = scaler.fit_transform(df[['Fare']])  # Optional: scale Fare too

    #Encoding categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['Sex', 'Embarked']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Sex', 'Embarked']))
    encoded_df.index = df.index
    
    X = pd.concat([df[['Age', 'Fare', 'Pclass']], encoded_df], axis=1)
    
    return train_test_split(X, y, test_size=0.3, random_state=1)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    train_and_evaluate(X_train, X_test, y_train, y_test)
