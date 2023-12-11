import pandas as pd
from joblib import load
from preprocess import process_new_data

def predict_insider_threat(model, features):
    # Prompt user for input
    user_id = input("Enter user ID: ")
    date = input("Enter date (YYYY-MM-DD HH:mm:ss): ")
    after_hours_attempt = input("Is the login attempt done after hours? (yes/no): ").lower() == 'yes'
    external_device_connected = input("Is an external device connected? (yes/no): ").lower() == 'yes'
    attachment_count = int(input("Enter the number of email attachments: "))

    # Create a DataFrame with the user input
    new_data = pd.DataFrame({
        'user': [user_id],
        'date': [date],
        'after_hours': [1 if after_hours_attempt else 0],
        'device_count': [1 if external_device_connected else 0],
        'attachment_count': [attachment_count],
    })

    try:
        # Process the new data
        new_data_processed = process_new_data(new_data)

        # Ensure that the features in 'new_data_processed' match the expected 'features'
        new_features = new_data_processed[features]

        # Handle NaN or missing values in new_features
        if new_features.isnull().values.any():
            print("Input data contains missing values. Please provide complete information.")
            return

        # Predict using the model
        new_predictions = model.predict(new_features)

        # Print the prediction
        if new_predictions[0] == 1:
            print("ALERT: Potential insider threat detected!")
        else:
            print("No threat detected.")

    except Exception as e:
        print(f"Error processing new data: {e}")

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
