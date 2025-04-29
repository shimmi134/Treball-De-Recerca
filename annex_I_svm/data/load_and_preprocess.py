import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    X = df.drop('target', axis='columns')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df.target
    return X_scaled, y
