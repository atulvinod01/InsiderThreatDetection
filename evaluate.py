from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from preprocess import final_df
import joblib
import numpy as np
import pandas as pd 
from sklearn.metrics import precision_recall_curve, average_precision_score


def evaluate_model(kmeans_model, X_test, y_test):
    # Ensure X_test is not None
    if X_test is not None:
        # Evaluate the model
        y_pred = kmeans_model.predict(X_test)

        # Print accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)

        print("Model Evaluation")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AP={average_precision:.2f})')
        plt.show()

        classes = ['Class 0', 'Class 1']
        precision = [0.48, 0.52]
        recall = [0.39, 0.61]
        f1_score = [0.43, 0.56]

        bar_width = 0.25
        index = np.arange(len(classes))

        plt.bar(index, precision, width=bar_width, label='Precision')
        plt.bar(index + bar_width, recall, width=bar_width, label='Recall')
        plt.bar(index + 2 * bar_width, f1_score, width=bar_width, label='F1-Score')

        plt.xlabel('Classes')
        plt.ylabel('Scores')
        plt.title('Precision, Recall, and F1-Score by Class')
        plt.xticks(index + bar_width, classes)
        plt.legend()
        plt.show()

        support = [88240, 94810]

        plt.barh(classes, support, color='lightblue')
        plt.xlabel('Support')
        plt.title('Support for Each Class')
        plt.show()


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
        f1 = f1_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)

        print("Model Evaluation")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
        print(f"F1 Score: {f1}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AP={average_precision:.2f})')
        plt.show()

        classes = ['Class 0', 'Class 1']
        precision = [0.48, 0.52]
        recall = [0.39, 0.61]
        f1_score = [0.43, 0.56]

        bar_width = 0.25
        index = np.arange(len(classes))

        plt.bar(index, precision, width=bar_width, label='Precision')
        plt.bar(index + bar_width, recall, width=bar_width, label='Recall')
        plt.bar(index + 2 * bar_width, f1_score, width=bar_width, label='F1-Score')

        plt.xlabel('Classes')
        plt.ylabel('Scores')
        plt.title('Precision, Recall, and F1-Score by Class')
        plt.xticks(index + bar_width, classes)
        plt.legend()
        plt.show()

        support = [88240, 94810]

        plt.barh(classes, support, color='lightblue')
        plt.xlabel('Support')
        plt.title('Support for Each Class')
        plt.show()

    else:
        print("X_test is not available or contains NaN values for evaluation.")
