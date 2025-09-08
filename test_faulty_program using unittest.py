# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 05:04:10 2025

@author: hongf
"""

import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# --- These simulate importing from your main script --- #
def load_data():
    data = load_iris(as_frame=True)
    df = data.frame  # Intentional bug: should be data.frame or data.data
    df['target'] = data.target
    return df

def preprocess(df):
    features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Bug: wrong column names
    X = features
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit(X)  # Bug: should be fit_transform
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# --- Now the unit tests --- #
class TestFaultyProgram(unittest.TestCase):

    def test_data_loading(self):
        """Test that data loads and contains expected columns."""
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = pd.DataFrame(data.data)
        df['target'] = data.target
        self.assertIn('target', df.columns)
        self.assertEqual(df.shape[0], 150)

    def test_column_names_exist(self):
        """Ensure the correct column names exist in iris dataset."""
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target
        # This will fail due to incorrect column names in `preprocess`
        expected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_scaling(self):
        """Ensure that scaling returns a transformed array."""
        data = load_iris(as_frame=True)
        df = pd.DataFrame(data.data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        self.assertEqual(X_scaled.shape, df.shape)

    def test_model_training(self):
        """Test that model can train on the correct data."""
        data = load_iris(as_frame=True)
        df = pd.DataFrame(data.data)
        df['target'] = data.target
        X_train, X_test, y_train, y_test = train_test_split(
            df.iloc[:, :-1], df['target'], test_size=0.2, random_state=42
        )
        acc = train_model(X_train, X_test, y_train, y_test)
        self.assertGreater(acc, 0.5)

if __name__ == '__main__':
    unittest.main()
