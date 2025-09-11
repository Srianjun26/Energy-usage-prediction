import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the test data
try:
    test_df = pd.read_csv('testdata.csv')
    # Drop the index column if it exists from saving the csv
    if 'Unnamed: 0' in test_df.columns:
        test_df = test_df.drop('Unnamed: 0', axis=1)

    # Store actual values for evaluation
    y_actual = test_df['Appliances']
    x_test_data = test_df.drop('Appliances', axis=1)
    
except FileNotFoundError:
    print("testdata.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Define filenames for the models
rf_model_filename = 'random_forest_regressor.pkl'
et_model_filename = 'extra_trees_regressor.pkl'

# Load and predict with Random Forest Regressor
try:
    rf_model = joblib.load(rf_model_filename)
    rf_predictions = rf_model.predict(x_test_data)
    
    # Evaluate Random Forest model
    rf_mse = mean_squared_error(y_actual, rf_predictions)
    rf_mae = mean_absolute_error(y_actual, rf_predictions)
    rf_r2 = r2_score(y_actual, rf_predictions)
    
    print("Random Forest Regressor Predictions:")
    print(rf_predictions)
    print(f"Random Forest Regression - MSE: {rf_mse}, MAE: {rf_mae}, R² Score: {rf_r2}\n")
    
    # Add predictions to the DataFrame for display
    test_df['RF_prediction'] = rf_predictions
    
except FileNotFoundError:
    print(f"Random Forest model file '{rf_model_filename}' not found.")
    
# Load and predict with Extra Trees Regressor
try:
    et_model = joblib.load(et_model_filename)
    et_predictions = et_model.predict(x_test_data)
    
    # Evaluate Extra Trees model
    et_mse = mean_squared_error(y_actual, et_predictions)
    et_mae = mean_absolute_error(y_actual, et_predictions)
    et_r2 = r2_score(y_actual, et_predictions)
    
    print("Extra Trees Regressor Predictions:")
    print(et_predictions)
    print(f"Extra Trees Regression - MSE: {et_mse}, MAE: {et_mae}, R² Score: {et_r2}\n")

    # Add predictions to the DataFrame for display
    test_df['ET_prediction'] = et_predictions

except FileNotFoundError:
    print(f"Extra Trees model file '{et_model_filename}' not found.")

# Display the DataFrame with all predictions
print("Test Data with Predictions:")
print(test_df)