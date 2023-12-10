# main.py
from preprocess import load_and_preprocess_data
from train import train_model
from monitoring import process_and_update_model
from predict import predict_insider_threat
from evaluate import evaluate_model, evaluate_trained_model
import time
import threading
import joblib

# Check if a trained model exists
model_filename = 'kmeans_model.joblib'
features = None
final_df = None
X_test = None
y_test = None

try:
    kmeans_model = joblib.load(model_filename)
    print("Loaded pre-existing model.")
except FileNotFoundError:
    # Load and preprocess data initially
    X, X_train, X_test, y_train, y_test, features, final_df = load_and_preprocess_data()

    # Train the KMeans model
    kmeans_model = train_model(X, model_filename)

# Evaluate the initial model
if X_test is not None:
    evaluate_model(kmeans_model, X_test, y_test)
else:
    evaluate_trained_model(model_filename, X_test, y_test)
    

# Start the monitoring loop in a separate thread
monitoring_thread = threading.Thread(target=process_and_update_model, args=(kmeans_model, features, final_df))
monitoring_thread.start()
monitoring_thread.join()

# Add a small delay to allow the monitoring thread to perform initial updates
time.sleep(2)

# Allow the user to interactively predict insider threats
'''while True:
    user_input = input("Do you want to predict an insider threat? (yes/no): ").lower()
    if user_input == 'yes':
        predict_insider_threat(kmeans_model, features)
    elif user_input == 'no':
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")'''
