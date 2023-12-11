import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
import pandas as pd
from joblib import load
from preprocess import load_and_preprocess_data

def evaluate_model(kmeans_model, X_test, y_test):
    # Ensure X_test is not None
    if X_test is not None:
        # Evaluate the model
        y_pred = kmeans_model.predict(X_test)

        # Print accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Model Evaluation")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
        print(f"F1 Score: {f1}")

        # Visualize clusters using scatter plot
        visualize_clusters_with_centroids(X_test, kmeans_model)
        
        # Plot the Elbow Method
        plot_elbow_method(X_test)
    else:
        print("X_test is not available for evaluation.")

def evaluate_trained_model(model_filename, X_test, y_test):
    # Load the pre-trained model
    kmeans_model = load(model_filename)

    # Ensure X_test is not None and does not contain NaN values
    if X_test is not None and not np.isnan(X_test).any():
        # Evaluate the model
        y_pred = kmeans_model.predict(X_test)

        # Print accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Model Evaluation")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
        print(f"F1 Score: {f1}")

        # Visualize clusters using scatter plot
        visualize_clusters_with_centroids(X_test, kmeans_model)

        # Plot the Elbow Method
        plot_elbow_method(X_test)
    else:
        print("X_test is not available or contains NaN values for evaluation.")

def visualize_clusters_with_centroids(X_test, kmeans_model):
    # Perform dimensionality reduction (PCA) if X_test has more than two features
    if X_test.shape[1] > 2:
        X_test_reduced = PCA(n_components=2).fit_transform(X_test)
    else:
        X_test_reduced = X_test

    # Add a 'cluster' column to the DataFrame
    X_test_clustered = X_test.copy()
    X_test_clustered['cluster'] = kmeans_model.predict(X_test)

    # Plot data points
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='day_of_week', y='url_count', hue='cluster', data=X_test_clustered, palette='viridis', s=100, alpha=0.8)

    # Plot centroids
    centroids_reduced = PCA(n_components=2).fit_transform(kmeans_model.cluster_centers_)
    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='X', s=200, c='red', label='Centroids')

    plt.title('Scatter Plot for Clusters with Centroids')
    plt.xlabel('Day of the Week')
    plt.ylabel('URL Count')
    plt.legend()

    plt.show()

def plot_elbow_method(X):
    distortions = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.plot(range(1, 11), distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

