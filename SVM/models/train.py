from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def train_model(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['linear', 'poly', 'rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    print('Best parameters:', grid.best_params_)
    print('Best estimator:', grid.best_estimator_)
    return grid.best_estimator_
