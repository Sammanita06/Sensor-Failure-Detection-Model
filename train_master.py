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

# 5. Preprocessor Setup (Constructed to execute in sequential stages)
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier_clipper', IQROutlierClipper(factor=1.5)),
    ('scaler', StandardScaler()),
])

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

# Sequential pipeline: Impute Raw -> Engineer Features -> Scale & Encode
feature_processing_pipeline = Pipeline([
    ('raw_imputer', ColumnTransformer(
        transformers=[
            ('num_imp', SimpleImputer(strategy='median'), raw_numeric_cols),
            ('ord_imp', SimpleImputer(strategy='most_frequent'), ordinal_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )),
    ('feature_engineer', MaintenanceFeatureEngineer()),
    ('scaler_encoder', ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, all_numeric_cols),
            ('ord', ordinal_pipeline, ordinal_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
        n_jobs=1
    ))
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

# 8. Calculate Permutation Feature Importances
print("Calculating Permutation Feature Importances...")
X_test_transformed = hgb_pipeline.named_steps['preprocessor'].transform(x_test)

try:
    feature_names = all_numeric_cols + ordinal_cols
except Exception:
    feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

classifier = hgb_pipeline.named_steps['classifier']

perm_result = permutation_importance(
    classifier,
    X_test_transformed,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_result.importances_mean,
    'Std': perm_result.importances_std,
}).sort_values(by='Importance', ascending=False)

print("=" * 60)
print("BASE PLAN EXPORT SUMMARY")
print("=" * 60)
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Accuracy:          {baseline_acc:.4f}")
print(f"F1-Score:          {baseline_f1:.4f}")
print(f"Precision:         {baseline_precision:.4f}")
print(f"Recall:            {baseline_recall:.4f}\n")

print("Top Feature Importances:")
print(importance_df.head().to_string(index=False))
print("=" * 60)

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
