# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=str, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)
    print(f"Model path: {args.model_path}")
    print(f"Model info output path: {args.model_info_output_path}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    mlflow.start_run()

    # Load model
    model = mlflow.sklearn.load_model(args.model_path)

    # Log model
    mlflow.sklearn.log_model(model, args.model_name)
    
    # Register the model directly from folder (MLflow expects MLmodel file inside) 
    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/{args.model_name}'
    print(f"Registering from URI: {model_uri}")
    
    mlflow_model = mlflow.register_model(model_uri, args.model_name)  # register the model with model_uri and model_name
    model_version = mlflow_model.version
    print(f"Registered model version: {mlflow_model.version}")

    # Write model info s JSON
    print("Writing JSON")
    model_info = {"id": f"{args.model_name}:{model_version}"}
    os.makedirs(args.model_info_output_path, exist_ok=True)
    output_path = os.path.join(args.model_info_output_path, "model_info.json")  # Specify the name of the JSON file (model_info.json)
    
    with open(output_path, "w") as of:
        json.dump(model_info, of)  # write model_info to the output file

    print(f"Model info written to: {output_path}")
    
    mlflow.end_run()

if __name__ == "__main__":
  
    # Parse Arguments
    args = parse_args()
    
    
    main(args)

    
