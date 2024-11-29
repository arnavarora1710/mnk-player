import pandas as pd
import torch
import numpy as np

def load_data(train_path='../data/train', test_path='../data/test'):
    train_data = pd.read_csv(f'{train_path}/train.csv')
    test_data = pd.read_csv(f'{test_path}/test.csv')

    # Select only numeric columns that exist in both datasets
    train_numeric = train_data.select_dtypes(include=['int64', 'float64']).columns
    test_numeric = test_data.select_dtypes(include=['int64', 'float64']).columns
    
    # Get common columns between train and test
    common_columns = list(set(train_numeric).intersection(set(test_numeric)))
    
    # Extract features (all common numeric columns except the last one which is target)
    feature_columns = common_columns[:-1]  # Assuming last numeric column is target
    target_column = common_columns[-1]
    
    # Get features and target
    X_train = train_data[feature_columns].astype(np.float32).values
    y_train = train_data[target_column].astype(np.float32).values

    X_test = test_data[feature_columns].astype(np.float32).values
    y_test = test_data[target_column].astype(np.float32).values

    # Print debug information
    print(f"Feature columns being used: {feature_columns}")
    print(f"Target column: {target_column}")
    print(f"Number of features: {len(feature_columns)}")

    features_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    features_tensor_test = torch.tensor(X_test, dtype=torch.float32)
    target_tensor_train = torch.tensor(y_train, dtype=torch.float32)
    target_tensor_test = torch.tensor(y_test, dtype=torch.float32)

    return features_tensor_train, target_tensor_train, features_tensor_test, target_tensor_test
