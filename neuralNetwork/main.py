import numpy as np
from model.train import train_model
from model.evaluate import evaluate_model
from data.load_and_preprocess_data import load_and_preprocess_data
from sklearn.model_selection import train_test_split

def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model, history = train_model(X_train, y_train)
    evaluate_model(model, history, X_test, y_test, X, y)

if __name__ == '__main__':
    main()
