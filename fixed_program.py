import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def load_data():
    data = load_iris(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df

def preprocess(df):
    features = df[['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']]
    X = features
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    return acc

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    train_model(X_train, X_test, y_train, y_test)
