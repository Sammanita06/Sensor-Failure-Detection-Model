import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """Clips numerical features based on Interquartile Range (IQR) bounds."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.select_dtypes(include=[np.number]).columns:
            q25 = X_df[col].quantile(0.25)
            q75 = X_df[col].quantile(0.75)
            iqr = q75 - q25
            lower = q25 - (self.factor * iqr)
            upper = q75 + (self.factor * iqr)
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df


class MaintenanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers operational telemetry feature ratios and interaction metrics."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        # Power & Heat Dissipation Features
        if 'Rotational speed [rpm]' in X_df.columns and 'Torque [Nm]' in X_df.columns:
            X_df['Power_kW'] = (X_df['Rotational speed [rpm]'] * X_df['Torque [Nm]']) / 9550.0

        if 'Air temperature [K]' in X_df.columns and 'Process temperature [K]' in X_df.columns:
            X_df['Temp_Difference'] = X_df['Process temperature [K]'] - X_df['Air temperature [K]']

        if 'Tool wear [min]' in X_df.columns and 'Torque [Nm]' in X_df.columns:
            X_df['Wear_Torque_Ratio'] = X_df['Tool wear [min]'] * X_df['Torque [Nm]']

        return X_df
