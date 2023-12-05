import pandas as pd
from joblib import load
from preprocess import process_new_data

def predict_insider_threat(model, features):
    # Prompt user for input
    user_id = input("Enter user ID: ")
    date = input("Enter date (YYYY-MM-DD HH:mm:ss): ")

    # Simulate checking if an external device is connected
    external_device_connected = input("Is an external device connected? (yes/no): ").lower() == 'yes'

    # Simulate checking if a mail sent has an attachment
    attachment_count = int(input("Enter the number of email attachments: "))

    # Create a DataFrame with the user input
    new_data = pd.DataFrame({
        'user': [user_id],
        'date': [date],
        'device_count': [1 if external_device_connected else 0],  # Feature engineering for device count
        'attachment_count': [attachment_count],  # Feature engineering for attachment count
        # Add other features based on your requirements
    })

    # Predict using the model
    new_features = new_data[features]
    new_predictions = model.predict(new_features)

    # Print the prediction
    if new_predictions[0] == 1:
        print("ALERT: Potential insider threat detected!")
    else:
        print("No threat detected.")

if __name__ == "__main__":
    # Ask if the user wants to predict an insider threat
    predict_request = input("Do you want to predict an insider threat? (yes/no): ").lower()

    if predict_request == 'yes':
        # Load the trained model
        trained_model = load('kmeans_model.joblib')  # Update with the correct model file

        # Features used for prediction (should match the features used during training)
        model_features = ['device_count', 'attachment_count']  # Update with your features

        # Call the prediction function
        predict_insider_threat(trained_model, model_features)
    else:
        print("No prediction performed. Exiting.")
