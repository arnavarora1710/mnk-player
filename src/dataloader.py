import pandas as pd
import torch

def load_data(train_path='data/train', test_path='data/test'):
    train_data = pd.read_csv(f'{train_path}/train.csv')
    test_data = pd.read_csv(f'{test_path}/test.csv')

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    features_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    features_tensor_test = torch.tensor(X_test, dtype=torch.float32)
    target_tensor_train = torch.tensor(y_train, dtype=torch.long)
    target_tensor_test = torch.tensor(y_test, dtype=torch.long)

    return features_tensor_train, target_tensor_train, features_tensor_test, target_tensor_test