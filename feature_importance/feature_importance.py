import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data['target'] = cancer.target
X = data.iloc[:, 0:30]
y = data.iloc[:, -1]
model = ExtraTreesClassifier()
model.fit(X, y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print("Menys importants:")
print(feat_importances.sort_values().head(10))
print("Més importants:")
print(feat_importances.sort_values(ascending=False).head(10))
feat_importances.nlargest(10).plot(kind='barh')
plt.figure(figsize=(20, 8))
plt.show()
