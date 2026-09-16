import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from custom_transformers import MaintenanceFeatureEngineer

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MODEL LOADING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

MASTER_DATASET_PATH = 'historical_master_data.csv'
PIPELINE_PATH = 'predictive_maintenance_pipeline.joblib'

@st.cache_resource
def load_pipeline_payload(filepath):
    """Cache model payload in memory to prevent repeated disk reads."""
    return joblib.load(filepath)

try:
    payload = load_pipeline_payload(PIPELINE_PATH)
    pipeline = payload['pipeline']
    DEFAULT_THRESHOLD = float(payload.get('default_threshold', 0.50))
    BASELINE_F1 = float(payload.get('baseline_f1', 0.0))
except Exception as e:
    st.error(f"Failed to load pipeline model. Ensure '{PIPELINE_PATH}' exists. Error: {e}")
    st.stop()

# ------------------------------------------------------------------------------
# 2. APPLICATION HEADER & SIDEBAR
# ------------------------------------------------------------------------------
st.title("⚙️ Equipment Failure Monitoring & Predictive Maintenance")
st.markdown("Production-grade inference pipeline with automated two-tier concept drift remediation.")

st.sidebar.header("🛠️ Pipeline Controls")
st.sidebar.markdown(f"**Loaded Model:** `Calibrated Soft Voting Ensemble`")
st.sidebar.markdown(f"**Default Threshold:** `{DEFAULT_THRESHOLD:.4f}`")
st.sidebar.markdown(f"**Baseline Test F1-Score:** `{BASELINE_F1:.4f}`")

uploaded_file = st.sidebar.file_uploader("Upload Sensor Batch Data (CSV)", type=["csv"])

# ------------------------------------------------------------------------------
# 3. BATCH INFERENCE & MONITORING ENGINE
# ------------------------------------------------------------------------------
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.subheader("📋 Batch Data Preview")
    st.dataframe(batch_df.head(5), use_container_width=True)

    with st.spinner("Executing feature engineering & adaptive inference pipeline..."):
        try:
            X_batch = batch_df.drop(columns=['Machine failure', 'Product ID', 'UDI'], errors='ignore')
            probabilities = pipeline.predict_proba(X_batch)[:, 1]

            operating_threshold = DEFAULT_THRESHOLD
            predictions = (probabilities >= operating_threshold).astype(int)

            processed_df = batch_df.copy()
            processed_df['Failure_Probability'] = probabilities
            processed_df['Predicted_Failure'] = predictions

        except Exception as err:
            st.error(f"Inference execution failed. Verify dataset schema. Error: {err}")
            st.stop()

    # --------------------------------------------------------------------------
    # 4. CONCEPT DRIFT & AUTOMATED RECOVERY ENGINE
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📉 Real-Time Model Performance & Remediation")

    if 'Machine failure' in batch_df.columns:
        y_true = batch_df['Machine failure']
        current_f1 = f1_score(y_true, predictions, zero_division=0)
        current_precision = precision_score(y_true, predictions, zero_division=0)
        current_recall = recall_score(y_true, predictions, zero_division=0)

        f1_delta = current_f1 - BASELINE_F1
        DRIFT_TOLERANCE = 0.05

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("Batch Baseline F1", f"{current_f1:.4f}", delta=f"{f1_delta:+.4f}")
        d_col2.metric("Batch Precision", f"{current_precision:.4f}")
        d_col3.metric("Batch Recall", f"{current_recall:.4f}")

        if current_f1 < (BASELINE_F1 - DRIFT_TOLERANCE):
            d_col4.error("⚠️ CONCEPT DRIFT DETECTED")
            st.warning(
                f"**Alert:** Current batch F1-Score (`{current_f1:.4f}`) dropped below baseline "
                f"(`{BASELINE_F1:.4f}`) by > 5%. Engaging Two-Tier Remediation Protocol."
            )

            # TIER 1: INSTANT RUNTIME RECALIBRATION
            st.markdown("### ⚡ Tier 1: Instant Runtime Recalibration")
            precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
            f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

            best_idx = np.argmax(f1_scores)
            tier1_threshold = float(thresholds[best_idx])
            tier1_f1 = float(f1_scores[best_idx])

            operating_threshold = tier1_threshold
            predictions = (probabilities >= operating_threshold).astype(int)
            processed_df['Predicted_Failure'] = predictions

            tier1_recall = recall_score(y_true, predictions, zero_division=0)

            st.info(
                f"**Tier 1 Execution complete:** Threshold shifted from `{DEFAULT_THRESHOLD:.4f}` to `{tier1_threshold:.4f}`. "
                f"F1-Score restored to `{tier1_f1:.4f}` | Restored Recall: `{tier1_recall:.4f}`."
            )

            # TIER 2: CONTINUOUS ONLINE LEARNING
            st.markdown("### 🔄 Tier 2: Continuous Online Learning Cycle")
            st.write("Append labeled batch to historical data and execute full pipeline retraining.")

            if st.button("🚀 Execute Tier 2 Pipeline Retraining"):
                with st.spinner("Appending batch data & retraining pipeline..."):
                    try:
                        if os.path.exists(MASTER_DATASET_PATH):
                            master_df = pd.read_csv(MASTER_DATASET_PATH)
                            updated_df = pd.concat([master_df, batch_df], ignore_index=True).drop_duplicates()
                        else:
                            updated_df = batch_df.copy()

                        updated_df.to_csv(MASTER_DATASET_PATH, index=False)

                        X_retrain = updated_df.drop(columns=['Machine failure', 'Product ID', 'UDI'], errors='ignore')
                        y_retrain = updated_df['Machine failure']

                        pipeline.fit(X_retrain, y_retrain)

                        retrained_probs = pipeline.predict_proba(X_batch)[:, 1]
                        retrained_preds = (retrained_probs >= 0.50).astype(int)
                        retrained_f1 = f1_score(y_true, retrained_preds, zero_division=0)

                        updated_payload = {
                            'pipeline': pipeline,
                            'default_threshold': 0.50,
                            'baseline_f1': retrained_f1,
                            'required_cols': X_retrain.columns.tolist()
                        }
                        joblib.dump(updated_payload, PIPELINE_PATH)
                        st.cache_resource.clear()

                        st.success(
                            f"✅ Tier 2 Retraining Successful! Model updated with {len(updated_df):,} total samples. "
                            f"New Baseline F1: `{retrained_f1:.4f}`."
                        )
                        st.rerun()

                    except Exception as retrain_err:
                        st.error(f"Tier 2 Retraining Failed: {retrain_err}")
        else:
            d_col4.success("✅ Model Performing Within Bounds")
    else:
        st.info("💡 Uploaded CSV does not contain ground-truth target labels (`Machine failure`). Running inference at active threshold.")

    # --------------------------------------------------------------------------
    # 5. EXECUTIVE KPI DASHBOARD
    # --------------------------------------------------------------------------
    st.markdown("---")
    total_samples = len(processed_df)
    predicted_failures = int(processed_df['Predicted_Failure'].sum())
    normal_operations = total_samples - predicted_failures
    failure_rate = (predicted_failures / total_samples) * 100 if total_samples > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Batch Sample Count", f"{total_samples:,}")
    col2.metric("Normal Operating Units", f"{normal_operations:,}")
    col3.metric("Flagged Machine Failures", f"{predicted_failures:,}", delta=f"{failure_rate:.2f}% Rate", delta_color="inverse")
    col4.metric("Active Threshold", f"{operating_threshold:.4f}")

    # --------------------------------------------------------------------------
    # 6. VISUALIZATIONS
    # --------------------------------------------------------------------------
    st.subheader("📊 Sensor Distributions & Anomaly Analysis")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        fig_pie = px.pie(
            names=['Normal', 'Failure Risk'],
            values=[normal_operations, predicted_failures],
            title="Batch Prediction Composition",
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with viz_col2:
        fig_hist = px.histogram(
            processed_df,
            x='Failure_Probability',
            nbins=30,
            title="Predicted Failure Probability Distribution",
            color='Predicted_Failure',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'Failure_Probability': 'Failure Probability'}
        )
        fig_hist.add_vline(
            x=operating_threshold, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Active Threshold ({operating_threshold:.4f})"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --------------------------------------------------------------------------
    # 7. EXPORT TABLE
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🚨 Flagged Anomaly Action Center")
    flagged_df = processed_df[processed_df['Predicted_Failure'] == 1]

    if len(flagged_df) > 0:
        st.dataframe(flagged_df, use_container_width=True)
        csv_data = flagged_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Flagged Failure Report (CSV)",
            data=csv_data,
            file_name="maintenance_anomalies_report.csv",
            mime="text/csv"
        )
    else:
        st.success("No machine failures flagged in this batch.")

else:
    st.info("👈 Please upload a batch CSV file in the sidebar to run predictions and view monitoring metrics.")
