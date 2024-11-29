import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class LGBMModel:
    def __init__(self):
        self.model = lgb.LGBMClassifier()
        self.label_encoder = LabelEncoder()

    def train(self, features, target):
        target_encoded = self.label_encoder.fit_transform(target)

        X_train, X_val, y_train, y_val = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f'Validation Accuracy: {accuracy:.2f}')

    def predict(self, features):
        predictions = self.model.predict(features)
        return self.label_encoder.inverse_transform(predictions)
