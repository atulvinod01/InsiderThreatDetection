# train.py
from sklearn.cluster import KMeans
import joblib

def train_model(X, model_filename='kmeans_model.joblib'):
    try:
        # Try loading an existing model
        kmeans_model = joblib.load(model_filename)
        print("Loaded pre-existing model.")
        
    except FileNotFoundError:
        # Train a new model if no existing model is found
        kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)  # You may need to adjust the number of clusters
        kmeans_model.fit(X)
        #joblib.dump(kmeans_model, model_filename)
        print("Trained and saved a new model.")
    
    return kmeans_model
