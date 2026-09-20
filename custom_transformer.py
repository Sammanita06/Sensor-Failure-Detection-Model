import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MaintenanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer domain-specific features for predictive maintenance.
    """
    def __init__(self):
        self.required_cols = [ 
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X).copy()

        if 'Process temperature [K]' in X_df.columns and 'Air temperature [K]' in X_df.columns:
            X_df['Temperature_Difference'] = X_df['Process temperature [K]'] - X_df['Air temperature [K]']

        if 'Rotational speed [rpm]' in X_df.columns and 'Torque [Nm]' in X_df.columns:
            X_df['Power_Product'] = X_df['Rotational speed [rpm]'] * X_df['Torque [Nm]']

        if 'Tool wear [min]' in X_df.columns and 'Torque [Nm]' in X_df.columns:
            X_df['Overstrain_Product'] = X_df['Tool wear [min]'] * X_df['Torque [Nm]']

        return X_df

    def get_feature_names_out(self, input_features=None):
        """Passes through input feature names and appends newly engineered column names."""
        if input_features is None:
            return None

        feature_names = list(input_features)
        new_cols = ['Temperature_Difference', 'Power_Product', 'Overstrain_Product']
        for col in new_cols:
            if col not in feature_names:
                feature_names.append(col)
        return np.array(feature_names, dtype=object)


class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clip extreme numerical outliers based on IQR.
    """
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
                self.lower_bounds_[col] = float(q25 - (self.factor * iqr))
                self.upper_bounds_[col] = float(q75 + (self.factor * iqr))
        elif isinstance(X, np.ndarray):
            for idx in range(X.shape[1]):
                q25 = np.percentile(X[:, idx], 25)
                q75 = np.percentile(X[:, idx], 75)
                iqr = q75 - q25
                self.lower_bounds_[idx] = float(q25 - (self.factor * iqr))
                self.upper_bounds_[idx] = float(q75 + (self.factor * iqr))
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            for col, lower in self.lower_bounds_.items():
                if col in X_df.columns:
                    upper = self.upper_bounds_[col]
                    X_df[col] = np.clip(X_df[col], lower, upper)
            return X_df
        elif isinstance(X, np.ndarray):
            X_arr = X.copy()
            for idx, lower in self.lower_bounds_.items():
                if idx < X_arr.shape[1]:
                    upper = self.upper_bounds_[idx]
                    X_arr[:, idx] = np.clip(X_arr[:, idx], lower, upper)
            return X_arr
        return X

    def get_feature_names_out(self, input_features=None):
        """Passes through input feature names without modification."""
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)
