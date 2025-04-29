from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
