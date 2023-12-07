import pandas as pd
from joblib import load
from preprocess import process_new_data

def predict_insider_threat(model, features):
    # Prompt user for input
    user_id = input("Enter user ID: ")

    # Prompt for date and check if login attempt is done after hours
    date = input("Enter date (YYYY-MM-DD HH:mm:ss): ")
    after_hours_attempt = input("Is the login attempt done after hours? (yes/no): ").lower() == 'yes'

    # Simulate checking if an external device is connected
    external_device_connected = input("Is an external device connected? (yes/no): ").lower() == 'yes'

    # Simulate checking if a mail sent has an attachment
    attachment_count = int(input("Enter the number of email attachments: "))

    # Create a DataFrame with the user input
    new_data = pd.DataFrame({
        'user': [user_id],
        'date': [date],
        'after_hours': [1 if after_hours_attempt else 0],  # Feature engineering for after hours attempt
        'device_count': [1 if external_device_connected else 0],  # Feature engineering for device count
        'attachment_count': [attachment_count],  # Feature engineering for attachment count
        # Add other features based on your requirements
    })

    # Process the new data (if needed)
    new_data_processed = process_new_data(new_data)

    # Ensure that the features in 'new_data_processed' match the expected 'features'
    new_features = new_data_processed[features]

    # Predict using the model
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
        model_features = ['after_hours', 'device_count', 'attachment_count']  # Update with your features

        # Call the prediction function
        predict_insider_threat(trained_model, model_features)
    else:
        print("No prediction performed. Exiting.")
