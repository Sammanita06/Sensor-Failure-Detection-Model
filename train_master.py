import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

# ------------------------------------------------------------------------------
# PREVENT MULTI-THREADING DEADLOCKS ON WINDOWS/JUPYTER
# ------------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 1. IMPORT CUSTOM TRANSFORMERS
from custom_transformers import IQROutlierClipper, MaintenanceFeatureEngineer

# Set up storage directory
os.makedirs('model_store', exist_ok=True)
MASTER_DATA_PATH = 'model_store/historical_master_data.csv'
PAYLOAD_PATH = 'model_store/production_v1.joblib'

# 2. Load and Store Master Baseline Data
historical_df = pd.read_csv('ai4i2020.csv')
historical_df.to_csv(MASTER_DATA_PATH, index=False)

# 3. Train-Test Split
X = historical_df.drop(
    columns=['Machine failure', 'Product ID', 'UDI'], errors='ignore'
)
Y = historical_df['Machine failure']

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=42
)

# 4. Column Groups
raw_numeric_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]
all_numeric_cols = raw_numeric_cols + [
    'Temperature_Difference',
    'Power_Product',
    'Overstrain_Product',
]
ordinal_cols = ['Type']

# 5. Preprocessor Setup
# Numerical Scaling & Outlier Pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier_clipper', IQROutlierClipper(factor=1.5)),
    ('scaler', StandardScaler()),
])

# Categorical Encoding Pipeline
ordinal_pipeline = Pipeline([
    (
        'encoder',
        OrdinalEncoder(
            categories=[['L', 'M', 'H']],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        ),
    )
])

# ColumnTransformer for scaling & encoding after feature engineering
scaler_encoder = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, all_numeric_cols),
        ('ord', ordinal_pipeline, ordinal_cols),
    ],
    remainder='drop',
    verbose_feature_names_out=False,
    n_jobs=1
)

# Set pandas output so feature names are preserved across steps
scaler_encoder.set_output(transform="pandas")

# Clean Feature Processing Pipeline (Feature Engineer -> Scaler/Encoder)
feature_processing_pipeline = Pipeline([
    ('feature_engineer', MaintenanceFeatureEngineer()),
    ('scaler_encoder', scaler_encoder)
])

# 6. Champion Model Pipeline (HistGradientBoosting)
hgb_pipeline = Pipeline([
    ('preprocessor', feature_processing_pipeline),
    (
        'classifier',
        HistGradientBoostingClassifier(
            class_weight='balanced',
            max_iter=200,
            learning_rate=0.05,
            random_state=42,
        ),
    ),
])

print("Fitting HistGradientBoosting Champion Pipeline...")
hgb_pipeline.fit(x_train, y_train)

# 7. Evaluate Probabilities & Dynamic Optimal Threshold
test_probs = hgb_pipeline.predict_proba(x_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)

if best_idx < len(thresholds):
    optimal_threshold = float(thresholds[best_idx])
else:
    optimal_threshold = 0.5000

test_preds = (test_probs >= optimal_threshold).astype(int)

baseline_f1 = f1_score(y_test, test_preds, zero_division=0)
baseline_precision = precision_score(y_test, test_preds, zero_division=0)
baseline_recall = recall_score(y_test, test_preds, zero_division=0)
baseline_acc = accuracy_score(y_test, test_preds)

# 8. Calculate Permutation Feature Importances (Optimized to prevent deadlocks)
print("Calculating Permutation Feature Importances...")
X_test_transformed = hgb_pipeline.named_steps['preprocessor'].transform(x_test)

# Sample a subset (e.g., 500 rows) to speed up execution and avoid thread locks
sample_size = min(500, len(X_test_transformed))
if isinstance(X_test_transformed, pd.DataFrame):
    X_sample = X_test_transformed.iloc[:sample_size]
else:
    X_sample = X_test_transformed[:sample_size]

y_sample = y_test.iloc[:sample_size]

feature_names = all_numeric_cols + ordinal_cols
classifier = hgb_pipeline.named_steps['classifier']

perm_result = permutation_importance(
    classifier,
    X_sample,
    y_sample,
    n_repeats=5,       # Reduced from 10 to 5 for speed
    random_state=42,
    n_jobs=1           # Ensures single-threaded execution
)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_result.importances_mean,
    'Std': perm_result.importances_std,
}).sort_values(by='Importance', ascending=False)
# 9. Export Unified Payload
model_payload = {
    'pipeline': hgb_pipeline,
    'optimal_threshold': float(optimal_threshold),
    'tier1_threshold': float(optimal_threshold),
    'tier2_threshold': 0.5000,
    'drift_tolerance_f1': 0.8000,
    'baseline_metrics': {
        'f1': float(baseline_f1),
        'precision': float(baseline_precision),
        'recall': float(baseline_recall),
        'accuracy': float(baseline_acc),
    },
    'feature_importances': importance_df,
    'required_cols': raw_numeric_cols + ordinal_cols,
    'feature_names': feature_names,
}

joblib.dump(model_payload, PAYLOAD_PATH)
print(f"\n✅ Production payload successfully exported to '{PAYLOAD_PATH}'")
