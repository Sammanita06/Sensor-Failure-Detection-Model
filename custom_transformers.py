import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MaintenanceFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            return X

        X = X.copy()

        # Fill missing values if present
        if 'Air temperature [K]' in X.columns and X['Air temperature [K]'].isna().sum() > 0:
            X['Air temperature [K]'] = X['Air temperature [K]'].fillna(X['Air temperature [K]'].mean())

        # Temperature Difference
        if 'Process temperature [K]' in X.columns and 'Air temperature [K]' in X.columns:
            X['Temperature_Difference'] = X['Process temperature [K]'] - X['Air temperature [K]']
        else:
            X['Temperature_Difference'] = 0.0

        # Power Product
        if 'Rotational speed [rpm]' in X.columns and 'Torque [Nm]' in X.columns:
            X['Power_Product'] = X['Torque [Nm]'] * X['Rotational speed [rpm]']
        else:
            X['Power_Product'] = 0.0

        # Overstrain Product
        if 'Torque [Nm]' in X.columns and 'Tool wear [min]' in X.columns:
            X['Overstrain_Product'] = X['Torque [Nm]'] * X['Tool wear [min]']
        else:
            X['Overstrain_Product'] = 0.0

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
