import numpy as np
import matplotlib.pyplot as plt
from data.load_and_preprocess import load_and_preprocess_data
from models.train import train_model

def main():
    X_pca = load_and_preprocess_data()
    labels, cluster_centers = train_model(X_pca)

    plt.figure(figsize=(8, 6))
    for i in range(2):
        plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Clúster {i}')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=200, c='black', label='Centroides')
    plt.xlabel('Component Principal 1')
    plt.ylabel('Component Principal 2')
    plt.title('Agrupació k-means sobre dades de càncer de mama')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
