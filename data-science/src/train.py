# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train_data
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test_data
    parser.add_argument("--model_output", type=str, help="Path of output model")  # Specify the type for model_output
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest')  # Specify the type and default value for n_estimators
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree')  # Specify the type and default value for max_depth

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Read train and test data from CSV
    train_path = Path(args.train_data) / "train.csv"
    test_path = Path(args.test_data) / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")


    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split the data into input(X) and output(y) 
    y_train = train_df['price']  # Specify the target column
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])

    # Initialize and train a RandomForest Regressor
    # Handle "None" string from sweep space
    max_depth = None if args.max_depth == 0 else int(args.max_depth)

    model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=max_depth, random_state=42)  # Provide the arguments for RandomForestRegressor
    model.fit(X_train, y_train)  # Train the model

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")  # Provide the model name
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Predict using the RandomForest Regressor on test data
    yhat_test = model.predict(X_test)  # Predict the test data

    # Compute and log mean squared error for test data
    mse = mean_squared_error(y_test, yhat_test)
    print(f"Mean Squared Error of RandomForest Regressor on test set : {mse:.2f}")
    mlflow.log_metric("MSE", float(mse))  # Log the MSE

    # Save the model
    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(model, args.model_output)  
    print(f"Best model location: {args.model_output}")


if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

