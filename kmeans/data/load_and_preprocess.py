from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

def load_and_preprocess_data():
    data = load_breast_cancer()
    X = data.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    return pca.fit_transform(X_scaled)
