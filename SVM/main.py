from sklearn.model_selection import train_test_split
from data.load_and_preprocess import load_and_preprocess_data
from models.train import train_model
from models.evaluate import evaluate_model

def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, X, y)

if __name__ == '__main__':
    main()
