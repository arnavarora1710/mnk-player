from dataloader import load_data
from models.lgbm import LGBMModel
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model():
    """
    Load data, train the LightGBM regression model, and evaluate it.
    """
    # Load data using the data loader
    features_train, target_train, features_test, target_test = load_data()
    
    # Convert PyTorch tensors to numpy arrays
    X_train = features_train.numpy()
    y_train = target_train.numpy().ravel()  # Flatten to 1D array
    X_test = features_test.numpy()
    y_test = target_test.numpy().ravel()  # Flatten to 1D array

    # Initialize the LGBM model
    model = LGBMModel(n_estimators=1000, learning_rate=0.01)
    
    # Train the model
    model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        early_stopping_rounds=50,
        verbose=True
    )
    
    # Make predictions on the validation set
    val_preds = model.predict(X_test)
    
    # Calculate RMSE on validation data
    rmse = np.sqrt(mean_squared_error(y_test, val_preds))
    print(f"Validation RMSE: {rmse:.4f}")

    return model, rmse

if __name__ == "__main__":
    # Train the model and retrieve metrics
    trained_model, validation_rmse = train_model()
    print(f"Model training completed. Validation RMSE: {validation_rmse:.4f}")
