import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin

class MaintenanceFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        elif hasattr(X, "shape"):
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)
        else:
            self.feature_names_in_ = np.array([], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            cols = getattr(self, "feature_names_in_", [f"x{i}" for i in range(X.shape[1])])
            X_df = pd.DataFrame(X, columns=cols).copy()

        if 'Process temperature [K]' in X_df.columns and 'Air temperature [K]' in X_df.columns:
            X_df['Temperature_Difference'] = X_df['Process temperature [K]'] - X_df['Air temperature [K]']

        if 'Rotational speed [rpm]' in X_df.columns and 'Torque [Nm]' in X_df.columns:
            X_df['Power_Product'] = X_df['Rotational speed [rpm]'] * X_df['Torque [Nm]']

        if 'Tool wear [min]' in X_df.columns and 'Torque [Nm]' in X_df.columns:
            X_df['Overstrain_Product'] = X_df['Tool wear [min]'] * X_df['Torque [Nm]']

        return X_df

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            base_names = list(input_features)
        elif hasattr(self, "feature_names_in_"):
            base_names = list(self.feature_names_in_)
        else:
            base_names = []

        new_cols = ['Temperature_Difference', 'Power_Product', 'Overstrain_Product']
        for col in new_cols:
            if col not in base_names:
                base_names.append(col)

        return np.array(base_names, dtype=object)


class IQROutlierClipper(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            for col in X.select_dtypes(include=[np.number]).columns:
                q25 = X[col].quantile(0.25)
                q75 = X[col].quantile(0.75)
                iqr = q75 - q25
                self.lower_bounds_[col] = float(q25 - (self.factor * iqr))
                self.upper_bounds_[col] = float(q75 + (self.factor * iqr))
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)
            for idx in range(X.shape[1]):
                q25 = np.percentile(X[:, idx], 25)
                q75 = np.percentile(X[:, idx], 75)
                iqr = q75 - q25
                self.lower_bounds_[idx] = float(q25 - (self.factor * iqr))
                self.upper_bounds_[idx] = float(q75 + (self.factor * iqr))

        self.n_features_in_ = len(self.feature_names_in_)
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
                if isinstance(idx, int) and idx < X_arr.shape[1]:
                    upper = self.upper_bounds_[idx]
                    X_arr[:, idx] = np.clip(X_arr[:, idx], lower, upper)
            return X_arr
        return X
