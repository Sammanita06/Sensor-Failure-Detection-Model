# drift_monitor.py
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def load_payload_safely(payload_path):
    """
    Safely loads a joblib payload after verifying file existence,
    checking for un-downloaded Git LFS pointers (<1KB), and handling corruption errors.
    """
    if not os.path.exists(payload_path):
        raise FileNotFoundError(f"Model file missing at: {payload_path}")

    # Check if the file is empty or too small (e.g., Git LFS pointer ~100 bytes)
    if os.path.getsize(payload_path) < 1024:
        raise ValueError(
            f"File at '{payload_path}' is too small ({os.path.getsize(payload_path)} bytes). "
            "It might be a Git LFS pointer file or corrupted file."
        )

    try:
        return joblib.load(payload_path)
    except (EOFError, pickle.UnpicklingError) as e:
        raise RuntimeError(f"Corrupted model file at '{payload_path}': {e}")


def run_two_tier_inference(
    batch_df, y_true=None, payload_path="predictive_maintenance_pipeline.joblib"
):
    """
    Executes two-tier inference with automated performance monitoring and fallback capabilities.
    """
    # 1. Safely load the payload model file
    try:
        payload = load_payload_safely(payload_path)
    except Exception as err:
        # Return fallback status dict so UI / application interface displays a structured error
        return {
            "metrics": None,
            "active_tier": "Error",
            "drift_alert": False,
            "error_message": str(err),
        }

    pipeline = payload["pipeline"]
    t1_thresh = payload.get(
        "tier1_threshold",
        payload.get("optimal_threshold", payload.get("default_threshold", 0.50)),
    )
    t2_thresh = payload.get("tier2_threshold", 0.50)
    f1_tolerance = payload.get("drift_tolerance_f1", 0.80)

    # 2. Get continuous failure probabilities
    probs = pipeline.predict_proba(batch_df)[:, 1]

    # 3. Execute Tier 1 Predictions (Strict Threshold)
    preds_t1 = (probs >= t1_thresh).astype(int)

    execution_info = {
        "active_tier": f"Tier 1 (High Precision - {t1_thresh:.2f})",
        "threshold_used": t1_thresh,
        "predictions": preds_t1,
        "probabilities": probs,
        "drift_alert": False,
        "metrics": None,
        "error_message": None,
    }

    # 4. Perform Performance Tracking if Ground Truth is provided
    if y_true is not None:
        batch_f1 = f1_score(y_true, preds_t1, zero_division=0)
        batch_prec = precision_score(y_true, preds_t1, zero_division=0)
        batch_rec = recall_score(y_true, preds_t1, zero_division=0)

        execution_info["metrics"] = {
            "f1": batch_f1,
            "precision": batch_prec,
            "recall": batch_rec,
        }

        # Check Concept Drift Trigger
        if batch_f1 < f1_tolerance:
            # Fall back to Tier 2 (Threshold 0.50)
            preds_t2 = (probs >= t2_thresh).astype(int)

            execution_info["active_tier"] = (
                f"Tier 2 (Fallback Safety - {t2_thresh:.2f})"
            )
            execution_info["threshold_used"] = t2_thresh
            execution_info["predictions"] = preds_t2
            execution_info["drift_alert"] = True

            # Recalculate metrics under Tier 2
            execution_info["metrics"]["f1"] = f1_score(
                y_true, preds_t2, zero_division=0
            )
            execution_info["metrics"]["precision"] = precision_score(
                y_true, preds_t2, zero_division=0
            )
            execution_info["metrics"]["recall"] = recall_score(
                y_true, preds_t2, zero_division=0
            )

    return execution_info
