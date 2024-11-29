import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

class LGBMModel:
    def __init__(self, n_estimators=1000, learning_rate=0.01, objective='regression', metric='rmse'):
        """
        Initialize the LGBM model with default parameters.
        """
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective=objective,
            metric=metric
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=None, early_stopping_rounds=50, verbose=False):
        """
        Train the LightGBM regression model with proper handling of early stopping.
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        - X_val: Validation features (optional)
        - y_val: Validation target (optional)
        - num_boost_round: Number of boosting rounds
        - early_stopping_rounds: Number of rounds for early stopping
        - verbose: Whether to display detailed logs
        """
        # If validation data is not provided, create a split
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Update n_estimators if num_boost_round is provided
        if num_boost_round is not None:
            self.model.set_params(n_estimators=num_boost_round)
        
        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds)] if early_stopping_rounds else None,
            # verbose=verbose
        )

        # Calculate RMSE on the validation set
        y_pred = self.model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f'Validation RMSE: {rmse:.4f}')

    def predict(self, features):
        """
        Predict target values using the trained model.
        
        Parameters:
        - features: Input features for prediction
        
        Returns:
        - Predicted values
        """
        return self.model.predict(features)
