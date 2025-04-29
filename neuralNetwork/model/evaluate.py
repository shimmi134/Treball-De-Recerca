import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, history, X_test, y_test, X, y):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.title('Accuracy i loss del model')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
