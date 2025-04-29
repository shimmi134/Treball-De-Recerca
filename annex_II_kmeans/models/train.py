from sklearn.cluster import KMeans

def train_model(X_pca):
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    return labels, kmeans.cluster_centers_
