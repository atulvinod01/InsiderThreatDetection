Insider Threat Detection

Overview

This project is designed to detect and predict insider threats using a KMeans clustering model. The system involves data preprocessing, model training, monitoring, and interactive predictions.

Project Structure

- `app/`: Main project directory containing the application code.
  - `preprocess.py`: Data preprocessing module.
  - `train.py`: Module for training the KMeans clustering model.
  - `monitoring.py`: Module for monitoring and updating the model.
  - `predict.py`: Module for interactive prediction of insider threats.
  - `evaluate.py`: Module for evaluating the model's performance.
  - `main.py`: Main script to execute the entire workflow.

Usage

Ensure you have the required dependencies installed. Use the following commands:
   cd path/to/your/folder
   pip install -r requirements.txt



Files and Modules

preprocess.py: Handles data loading, preprocessing, and feature engineering.
train.py: Manages the training of the KMeans clustering model.
monitoring.py: Monitors data for updates and re-trains the model accordingly.
predict.py: Allows users to interactively predict insider threats.
evaluate.py: Evaluates the performance of the model.

Dependencies

pandas==1.3.3
scikit-learn==0.24.2
joblib==1.1.0
numpy==1.21.2
matplotlib==3.4.3
threading==0.1
scipy==1.7.2


