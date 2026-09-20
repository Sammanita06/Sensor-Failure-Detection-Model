import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def run_two_tier_inference(batch_df, y_true=None, payload_path='model_store/production_v1.joblib'):
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Model payload file '{payload_path}' not found.")
        
    payload = joblib.load(payload_path)
    pipeline = payload['pipeline']
    t1_thresh = payload.get('tier1_threshold', payload.get('optimal_threshold', 0.50))
    t2_thresh = payload.get('tier2_threshold', 0.50)
    f1_tolerance = payload.get('drift_tolerance_f1', 0.80)

    probs = pipeline.predict_proba(batch_df)[:, 1]
    preds_t1 = (probs >= t1_thresh).astype(int)
    
    execution_info = {
        'active_tier': 'Tier 1 (High Precision)',
        'threshold_used': t1_thresh,
        'predictions': preds_t1,
        'probabilities': probs,
        'drift_alert': False,
        'metrics': None
    }

    if y_true is not None:
        batch_f1 = f1_score(y_true, preds_t1, zero_division=0)
        batch_prec = precision_score(y_true, preds_t1, zero_division=0)
        batch_rec = recall_score(y_true, preds_t1, zero_division=0)

        execution_info['metrics'] = {'f1': batch_f1, 'precision': batch_prec, 'recall': batch_rec}

        if batch_f1 < f1_tolerance:
            preds_t2 = (probs >= t2_thresh).astype(int)
            
            execution_info['active_tier'] = 'Tier 2 (Fallback Safety)'
            execution_info['threshold_used'] = t2_thresh
            execution_info['predictions'] = preds_t2
            execution_info['drift_alert'] = True
            
            execution_info['metrics']['f1'] = f1_score(y_true, preds_t2, zero_division=0)
            execution_info['metrics']['recall'] = recall_score(y_true, preds_t2, zero_division=0)

    return execution_info
