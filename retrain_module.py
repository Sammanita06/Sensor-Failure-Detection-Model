import os
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Primary paths
MASTER_DATA_PATH = os.path.join(BASE_DIR, 'model_store', 'historical_master_data.csv')
PAYLOAD_PATH = os.path.join(BASE_DIR, 'model_store', 'production_v1.joblib')

# Fallback root path check for Streamlit Community Cloud deployments
ALT_PAYLOAD_PATH = os.path.join(BASE_DIR, 'predictive_maintenance_pipeline.joblib')

def execute_tier2_retrain(new_batch_df):
    """
    Appends new batch telemetry records, deduplicates, retrains the primary ML pipeline,
    re-evaluates drift performance, and updates stored joblib artifacts.
    """
    os.makedirs(os.path.dirname(MASTER_DATA_PATH), exist_ok=True)

    # 1. Load or Bootstrap Master Dataset
    if os.path.exists(MASTER_DATA_PATH):
        master_df = pd.read_csv(MASTER_DATA_PATH)
    else:
        # Self-healing fallback: Initialize master data storage from incoming batch
        master_df = new_batch_df.copy()

    # 2. Load Pipeline Payload Artifact
    active_payload_path = PAYLOAD_PATH if os.path.exists(PAYLOAD_PATH) else ALT_PAYLOAD_PATH
    
    if not os.path.exists(active_payload_path):
        raise FileNotFoundError(
            f"Unable to locate model payload artifact at '{PAYLOAD_PATH}' or '{ALT_PAYLOAD_PATH}'."
        )

    payload = joblib.load(active_payload_path)

    # Extract pipeline depending on payload structure type (dict vs raw estimator)
    if isinstance(payload, dict):
        pipeline = payload['pipeline']
        active_threshold = payload.get('optimal_threshold', payload.get('tier1_threshold', 0.50))
    else:
        pipeline = payload
        payload = {}
        active_threshold = 0.50

    # 3. Append & Deduplicate by UDI or Exact Row Vectors
    combined_df = pd.concat([master_df, new_batch_df], ignore_index=True)
    if 'UDI' in combined_df.columns:
        combined_df.drop_duplicates(subset=['UDI'], keep='last', inplace=True)
    else:
        combined_df.drop_duplicates(inplace=True)

    # Persist updated master historical storage back to disk
    combined_df.to_csv(MASTER_DATA_PATH, index=False)

    # 4. Prepare Retraining Splits
    X_updated = combined_df.drop(columns=['Machine failure', 'Product ID', 'UDI'], errors='ignore')
    Y_updated = combined_df['Machine failure']

    # Handle stratified split edge-case for single-class batches
    if Y_updated.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_updated, Y_updated, test_size=0.3, stratify=Y_updated, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_updated, Y_updated, test_size=0.3, random_state=42
        )

    # 5. Retrain Complete Pipeline
    pipeline.fit(X_train, y_train)

    # 6. Evaluate Updated Metrics
    new_probs = pipeline.predict_proba(X_test)[:, 1]
    new_preds = (new_probs >= active_threshold).astype(int)

    updated_f1 = f1_score(y_test, new_preds, zero_division=0)
    updated_precision = precision_score(y_test, new_preds, zero_division=0)
    updated_recall = recall_score(y_test, new_preds, zero_division=0)
    updated_acc = accuracy_score(y_test, new_preds)

    # 7. Permutation Feature Importances (Optimized Sub-sampling)
    print("Calculating Permutation Feature Importances...")
    
    # Process features through intermediate transformer steps safely
    X_test_transformed = X_test.copy()
    if 'feature_engineer' in pipeline.named_steps:
        X_test_transformed = pipeline.named_steps['feature_engineer'].transform(X_test_transformed)
    if 'preprocessor' in pipeline.named_steps:
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test_transformed)

    # Sub-sample evaluation set to prevent CPU memory spikes/hangs
    sample_size = min(500, len(X_test_transformed))
    if isinstance(X_test_transformed, pd.DataFrame):
        X_sample = X_test_transformed.iloc[:sample_size]
    else:
        X_sample = X_test_transformed[:sample_size]

    y_sample = y_test.iloc[:sample_size]

    classifier = pipeline.named_steps.get('classifier', pipeline[-1])
    
    # Retrieve feature names safely
    feature_names = payload.get(
        'feature_names', 
        [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
    )

    perm_result = permutation_importance(
        classifier,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        n_jobs=1
    )

    feature_imp_df = pd.DataFrame({
        'Feature': feature_names[:X_test_transformed.shape[1]],
        'Importance': perm_result.importances_mean,
        'Std': perm_result.importances_std,
    }).sort_values(by='Importance', ascending=False)

    # 8. Overwrite Stored Artifacts
    updated_payload = {
        'pipeline': pipeline,
        'optimal_threshold': float(active_threshold),
        'tier1_threshold': payload.get('tier1_threshold', active_threshold),
        'tier2_threshold': payload.get('tier2_threshold', 0.50),
        'drift_tolerance_f1': payload.get('drift_tolerance_f1', 0.80),
        'baseline_metrics': {
            'f1': float(updated_f1),
            'precision': float(updated_precision),
            'recall': float(updated_recall),
            'accuracy': float(updated_acc)
        },
        'feature_importances': feature_imp_df,
        'required_cols': payload.get('required_cols', X_updated.columns.tolist()),
        'feature_names': feature_names
    }

    # Save to both target locations to maintain Cloud runtime persistence
    joblib.dump(updated_payload, active_payload_path)
    if os.path.exists(PAYLOAD_PATH) and active_payload_path != PAYLOAD_PATH:
        joblib.dump(updated_payload, PAYLOAD_PATH)

    return {
        'total_master_records': len(combined_df),
        'new_records_added': len(combined_df) - len(master_df),
        'updated_f1': updated_f1,
        'updated_precision': updated_precision,
        'updated_recall': updated_recall
    }
