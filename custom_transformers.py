import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MaintenanceFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        # 1. Temperature Ratio
        if 'Process temperature [K]' in X.columns and 'Air temperature [K]' in X.columns:
            X['Temp_Ratio'] = X['Process temperature [K]'] / (X['Air temperature [K]'] + 1e-6)
        else:
            X['Temp_Ratio'] = 0.0

        # 2. Power Proxy & Strain Ratio
        if 'Rotational speed [rpm]' in X.columns and 'Torque [Nm]' in X.columns:
            X['Power_Proxy'] = X['Torque [Nm]'] * X['Rotational speed [rpm]']
            X['Strain_Ratio'] = X['Torque [Nm]'] / (X['Rotational speed [rpm]'] + 1e-6)
        else:
            X['Power_Proxy'] = 0.0
            X['Strain_Ratio'] = 0.0

        # 3. Power to Wear Ratio
        if 'Power_Proxy' in X.columns and 'Tool wear [min]' in X.columns:
            X['Power_To_Wear_Ratio'] = X['Power_Proxy'] / (X['Tool wear [min]'] + 1e-6)
        else:
            X['Power_To_Wear_Ratio'] = 0.0

        # 4. Thermal-Wear Interaction
        if 'Temp_Ratio' in X.columns and 'Tool wear [min]' in X.columns:
            X['Thermal_Wear_Interaction'] = X['Temp_Ratio'] * X['Tool wear [min]']
        else:
            X['Thermal_Wear_Interaction'] = 0.0

        return X

class IQROutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=[np.number]).columns:
                q25 = X[col].quantile(0.25)
                q75 = X[col].quantile(0.75)
                iqr = q75 - q25
                self.lower_bounds_[col] = q25 - (self.factor * iqr)
                self.upper_bounds_[col] = q75 + (self.factor * iqr)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        X = X.copy()
        for col, lower in self.lower_bounds_.items():
            if col in X.columns:
                upper = self.upper_bounds_[col]
                X[col] = np.clip(X[col], lower, upper)
        return X
