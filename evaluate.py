# evaluate.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np

def evaluate_model(kmeans_model, X_test, y_test):
    # Ensure X_test is not None
    if X_test is not None:
        # Evaluate the model
        y_pred = kmeans_model.predict(X_test)

        # Print accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print("Model Evaluation")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
    else:
        print("X_test is not available for evaluation.")

def evaluate_trained_model(model_filename, X_test, y_test):
    # Load the pre-trained model
    kmeans_model = joblib.load(model_filename)

    # Ensure X_test is not None and does not contain NaN values
    if X_test is not None and not np.isnan(X_test).any():
        # Evaluate the model
        y_pred = kmeans_model.predict(X_test)

        # Print accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print("Model Evaluation")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
    else:
        print("X_test is not available or contains NaN values for evaluation.")
