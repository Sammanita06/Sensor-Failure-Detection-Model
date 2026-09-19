import os
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

MASTER_DATA_PATH = 'model_store/historical_master_data.csv'
PAYLOAD_PATH = 'model_store/production_v1.joblib'

def execute_tier2_retrain(new_batch_df):
    """
    Appends new batch data to master historical CSV without duplicates,
    retrains the production pipeline safely, evaluates performance, 
    and updates production_v1.joblib.
    """
    if not os.path.exists(MASTER_DATA_PATH) or not os.path.exists(PAYLOAD_PATH):
        raise FileNotFoundError("Master storage files missing. Run initial setup script first.")

    # 1. Load Master Dataset and Existing Payload
    master_df = pd.read_csv(MASTER_DATA_PATH)
    payload = joblib.load(PAYLOAD_PATH)
    pipeline = payload['pipeline']
    active_threshold = payload.get('optimal_threshold', payload.get('tier1_threshold', 0.50))

    # 2. Append & Deduplicate by UDI or exact rows
    combined_df = pd.concat([master_df, new_batch_df], ignore_index=True)
    if 'UDI' in combined_df.columns:
        combined_df.drop_duplicates(subset=['UDI'], keep='last', inplace=True)
    else:
        combined_df.drop_duplicates(inplace=True)

    # Save expanded historical dataset back to disk
    combined_df.to_csv(MASTER_DATA_PATH, index=False)

    # 3. Prepare Retraining Splits (Explicitly drop non-feature IDs like UDI and Product ID)
    X_updated = combined_df.drop(columns=['Machine failure', 'Product ID', 'UDI'], errors='ignore')
    Y_updated = combined_df['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X_updated, Y_updated, test_size=0.3, stratify=Y_updated, random_state=42
    )

    # 4. Retrain Complete Pipeline
    pipeline.fit(X_train, y_train)

    # 5. Evaluate Updated Baseline Metrics
    new_probs = pipeline.predict_proba(X_test)[:, 1]
    new_preds = (new_probs >= active_threshold).astype(int)

    updated_f1 = f1_score(y_test, new_preds, zero_division=0)
    updated_precision = precision_score(y_test, new_preds, zero_division=0)
    updated_recall = recall_score(y_test, new_preds, zero_division=0)
    updated_acc = accuracy_score(y_test, new_preds)

    # 6. Recalculate Updated Permutation Importance
    feature_engineer = pipeline.named_steps['feature_engineer']
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # Step-by-step transformation sequence to ensure engineered columns exist
    X_test_engineered = feature_engineer.transform(X_test)
    X_test_trans = preprocessor.transform(X_test_engineered)

    perm_res = permutation_importance(
        classifier, X_test_trans, y_test, scoring='f1', n_repeats=3, random_state=42, n_jobs=1
    )
    
    feature_names = payload.get('feature_names', [f"feature_{i}" for i in range(X_test_trans.shape[1])])
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names[:X_test_trans.shape[1]],
        'Importance': perm_res.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # 7. Overwrite Payload Artifacts
    updated_payload = {
        'pipeline': pipeline,
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
        'feature_names': X_updated.columns.tolist()
    }

    joblib.dump(updated_payload, PAYLOAD_PATH)

    return {
        'total_master_records': len(combined_df),
        'new_records_added': len(combined_df) - len(master_df),
        'updated_f1': updated_f1,
        'updated_precision': updated_precision,
        'updated_recall': updated_recall
    }
