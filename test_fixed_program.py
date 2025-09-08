import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from fixed_program import preprocess, train_model

class TestFixedProgram(unittest.TestCase):

    def test_load_data_shape(self):
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target
        self.assertEqual(df.shape, (150, 5))

    def test_preprocess_output(self):
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target
        X_train, X_test, y_train, y_test = preprocess(df)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

    def test_model_accuracy(self):
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target
        X_train, X_test, y_train, y_test = preprocess(df)
        acc = train_model(X_train, X_test, y_train, y_test)
        self.assertGreater(acc, 0.5)

if __name__ == '__main__':
    unittest.main()
