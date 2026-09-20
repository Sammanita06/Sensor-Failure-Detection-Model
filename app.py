import os
import sys
import types
import io
import pickle

# -----------------------------------------------------------------------------
# PATH & MODULE INJECTION FIX FOR STREAMLIT CLOUD / JOBLIB UNPICKLING
# -----------------------------------------------------------------------------
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# Import custom transformers
try:
    import custom_transformers
    from custom_transformers import IQROutlierClipper, MaintenanceFeatureEngineer
except ImportError:
    # Build dynamic fallback module if pathing fails during Cloud initialization
    custom_transformers = types.ModuleType("custom_transformers")
    sys.modules["custom_transformers"] = custom_transformers
    from custom_transformers import IQROutlierClipper, MaintenanceFeatureEngineer

# Register classes across all potential pickle import namespaces
import __main__
setattr(__main__, "IQROutlierClipper", IQROutlierClipper)
setattr(__main__, "MaintenanceFeatureEngineer", MaintenanceFeatureEngineer)

setattr(sys.modules["custom_transformers"], "IQROutlierClipper", IQROutlierClipper)
setattr(sys.modules["custom_transformers"], "MaintenanceFeatureEngineer", MaintenanceFeatureEngineer)

sys.modules["IQROutlierClipper"] = IQROutlierClipper
sys.modules["MaintenanceFeatureEngineer"] = MaintenanceFeatureEngineer

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import shap
import streamlit as st
from sklearn.inspection import permutation_importance

# Custom operational modules
from drift_monitor import run_two_tier_inference
from retrain_module import execute_tier2_retrain
from logger_db import fetch_historical_logs, log_batch_execution

# -----------------------------------------------------------------------------
# CUSTOM UNPICKLER OVERRIDE
# -----------------------------------------------------------------------------
class SafeCustomUnpickler(pickle.Unpickler):
    """Custom unpickler that intercepts missing class path lookups during deserialization."""
    def find_class(self, module, name):
        if name in ["IQROutlierClipper", "MaintenanceFeatureEngineer"]:
            return globals()[name]
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            if hasattr(custom_transformers, name):
                return getattr(custom_transformers, name)
            if hasattr(__main__, name):
                return getattr(__main__, name)
            raise

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & DEEP NAVY BLUE / GRID CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Machinery Diagnostics & Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast CSS with Tech Grid, Solid Dark Slate, and Cyan Accents
st.markdown(
    """
    <style>
    /* Dark Mode Solid & Subtle Tech Grid Pattern */
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(rgba(56, 189, 248, 0.07) 1px, transparent 0);
        background-size: 24px 24px;
        color: #f8fafc;
    }

    /* Typography & High-Contrast Readability Rules */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700;
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    .stCaption, label, div[data-baseweb="label"], p, span {
        color: #e2e8f0 !important;
        font-weight: 500;
    }

    code, .mono-text, div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }

    /* Card-Based Grid Container Styling */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    }

    /* Production-Ready Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 8px 10px 0px 10px;
        border-radius: 12px 12px 0px 0px;
        border-bottom: 2px solid #334155;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #0f172a;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #94a3b8 !important;
        border: 1px solid #334155;
        border-bottom: none;
        transition: all 0.2s ease-in-out;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155;
        color: #f8fafc !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-top: 3px solid #38bdf8 !important;
        border-left: 1px solid #334155 !important;
        border-right: 1px solid #334155 !important;
        border-bottom: 2px solid #1e293b !important;
    }

    /* Form Fields */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] * {
        color: #f8fafc !important;
        background-color: #0f172a !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #0284c7;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #38bdf8;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #0369a1;
        box-shadow: 0px 0px 14px rgba(56, 189, 248, 0.4);
    }

    div[data-testid="stMetricValue"] {
        font-size: 30px;
        font-weight: 700;
        color: #38bdf8 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1. LOAD MODEL PIPELINE & TWO-TIER PAYLOAD
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_payload():
    # 1. Primary path matching root repo structure from screenshot
    model_path = os.path.join(app_dir, 'predictive_maintenance_pipeline.joblib')
    
    # 2. Fallback check for alternative folder pathing
    if not os.path.exists(model_path):
        model_path = os.path.join(app_dir, 'model_store', 'production_v1.joblib')
        
    if not os.path.exists(model_path):
        st.error(f"❌ Model payload not found. Please verify 'predictive_maintenance_pipeline.joblib' exists in root.")
        st.stop()
    
    try:
        payload = joblib.load(model_path)
    except (ImportError, AttributeError, ModuleNotFoundError):
        # Fallback to custom safe unpickler if standard module resolution fails
        with open(model_path, 'rb') as f:
            payload = SafeCustomUnpickler(f).load()
            
    return payload

payload = load_model_payload()

# Extract components from payload
if isinstance(payload, dict):
    pipeline = payload.get('pipeline')
    tier1_threshold = payload.get('tier1_threshold', 0.8900)
    tier2_threshold = payload.get('tier2_threshold', 0.5000)
    drift_tolerance_f1 = payload.get('drift_tolerance_f1', 0.8000)
    baseline_metrics = payload.get('baseline_metrics', {})
    feature_importance_df = payload.get('feature_importances', None)
else:
    pipeline = payload
    tier1_threshold = 0.8900
    tier2_threshold = 0.5000
    drift_tolerance_f1 = 0.8000
    feature_importance_df = None

# -----------------------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ SYSTEM STATUS")
    st.caption("TELEMETRY ENGINE V2.4")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Threshold Configs")
    st.markdown(f"`TIER-1 CUTOFF:` **{tier1_threshold:.4f}**")
    st.markdown(f"`TIER-2 FALLBACK:` **{tier2_threshold:.4f}**")
    st.markdown(f"`DRIFT TOLERANCE:` **{drift_tolerance_f1:.2f} F1**")
    
    st.markdown("---")
    st.markdown("#### 🟢 System Health Flag")
    st.success("MODEL PIPELINE ACTIVE & INLINE")

# -----------------------------------------------------------------------------
# SPLIT-SCREEN HERO SECTION
# -----------------------------------------------------------------------------
hero_left, hero_right = st.columns([2, 1])

with hero_left:
    st.title("⚡ Machinery Diagnostics & Predictive Telemetry Dashboard")
    st.markdown(
        "<span style='color: #38bdf8; font-weight: bold;'>SYSTEM STATUS: ONLINE</span> — Real-time sensor stream evaluations, "
        "batch inference processing, drift monitoring, and local/global SHAP model explainability.",
        unsafe_allow_html=True
    )

with hero_right:
    with st.container(border=True):
        st.caption("SYSTEM METADATA")
        st.markdown("`ENGINE:` Random Forest Pipeline")
        st.markdown("`INGESTION:` 6-Vector Sensor Stream")
        st.markdown("`INFERENCE:` Two-Tier Adaptive Guard")

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🛠️ Single Machine Diagnostic", 
    "📁 Batch Diagnostics", 
    "📊 Model Interpretability & SHAP",
    "📈 MLOps Model Decay Timeline"
])

# -----------------------------------------------------------------------------
# TAB 1: SINGLE MACHINE DIAGNOSTIC
# -----------------------------------------------------------------------------
with tab1:
    st.header("Real-Time Machine Operational Diagnostic")
    st.write("Adjust hardware sensor operational parameters to execute real-time fault detection.")

    with st.container(border=True):
        st.caption("SENSOR INPUT STREAM METRICS")
        col1, col2, col3 = st.columns(3)

        with col1:
            type_val = st.selectbox("Product Grade/Type", options=['L', 'M', 'H'], index=0)
            air_temp = st.number_input("Air Temperature [K]", min_value=250.0, max_value=350.0, value=300.0, step=0.1)

        with col2:
            process_temp = st.number_input("Process Temperature [K]", min_value=250.0, max_value=360.0, value=310.0, step=0.1)
            rot_speed = st.number_input("Rotational Speed [rpm]", min_value=500, max_value=3000, value=1500, step=10)

        with col3:
            torque = st.number_input("Torque Output [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
            tool_wear = st.number_input("Tool Cumulative Wear [min]", min_value=0, max_value=300, value=100, step=1)

    input_df = pd.DataFrame([{
        'Type': type_val,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rot_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }])

    st.markdown("---")
    
    if st.button("RUN TELEMETRY DIAGNOSTIC", type="primary"):
        probs = pipeline.predict_proba(input_df)[0]
        failure_prob = probs[1]
        is_failure = failure_prob >= tier1_threshold

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            with st.container(border=True):
                st.caption("PREDICTION TELEMETRY OUTPUT")
                st.metric(
                    label="Failure Risk Index", 
                    value=f"{failure_prob * 100:.2f}%", 
                    delta=f"Threshold: {tier1_threshold * 100:.1f}%",
                    delta_color="inverse" if is_failure else "normal"
                )
                
                if is_failure:
                    st.error("⚠️ **CRITICAL: FAULT PREDICTED!** Immediate Maintenance Required.")
                else:
                    st.success("✅ **SYSTEM NOMINAL:** Operating within safety limits.")

        with res_col2:
            with st.container(border=True):
                st.subheader("Local Model Explanation (SHAP Waterfall)")
                try:
                    preprocessor = pipeline.named_steps['preprocessor']
                    feature_engineer = pipeline.named_steps.get('feature_engineer', None)
                    classifier = pipeline.named_steps['classifier']

                    if feature_engineer:
                        x_engineered = feature_engineer.transform(input_df)
                    else:
                        x_engineered = input_df.copy()

                    x_trans = preprocessor.transform(x_engineered)

                    explainer = shap.Explainer(classifier)
                    shap_values = explainer(x_trans)

                    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e293b')
                    ax.set_facecolor('#1e293b')
                    plt.rcParams['text.color'] = '#f8fafc'
                    plt.rcParams['axes.labelcolor'] = '#f8fafc'
                    plt.rcParams['xtick.color'] = '#f8fafc'
                    plt.rcParams['ytick.color'] = '#f8fafc'
                    
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"SHAP local breakdown unavailable for this configuration: {e}")

# -----------------------------------------------------------------------------
# TAB 2: BATCH PREDICTIONS & DRIFT MONITORING
# -----------------------------------------------------------------------------
with tab2:
    st.header("Batch CSV Ingestion & Drift Controller")
    st.write("Upload machine log records (.csv) to perform batch diagnostics, evaluate concept drift, and record telemetry.")

    uploaded_file = st.file_uploader("Upload Sensor Diagnostic Logs (CSV)", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        with st.container(border=True):
            st.caption("BATCH STREAM DATA PREVIEW")
            st.dataframe(batch_df.head(), use_container_width=True)

        y_true = batch_df['Machine failure'] if 'Machine failure' in batch_df.columns else None

        results = run_two_tier_inference(batch_df, y_true=y_true)

        if results['metrics']:
            log_batch_execution(
                batch_size=len(batch_df),
                f1=results['metrics']['f1'],
                precision=results['metrics']['precision'],
                recall=results['metrics']['recall'],
                tier_used=results['active_tier'],
                drift_detected=results['drift_alert']
            )

        st.markdown("---")
        st.subheader("Drift Monitoring & Operational Tier Status")

        if results['drift_alert']:
            st.error(f"🚨 **CONCEPT DRIFT ALERT DETECTED!** Operating mode overridden to **{results['active_tier']}**.")
            st.warning(f"Batch performance F1 dropped below tolerance ({drift_tolerance_f1:.2f}). Tier 2 fallback engaged to prevent uncaptured failures.")
        else:
            st.success(f"✅ **NORMAL TELEMETRY OPERATING STATE:** Active Tier = **{results['active_tier']}**")

        if results['metrics']:
            with st.container(border=True):
                st.caption("EVALUATION METRICS FOR CURRENT INGESTED BATCH")
                col1, col2, col3 = st.columns(3)
                col1.metric("Batch F1-Score", f"{results['metrics']['f1']:.4f}")
                col2.metric("Batch Precision", f"{results['metrics']['precision']:.4f}")
                col3.metric("Batch Recall", f"{results['metrics']['recall']:.4f}")

            with st.expander("📊 View Live Feature Importance Spectrum (Ingested Batch)"):
                try:
                    X_batch = batch_df.drop(columns=['Machine failure', 'Product ID', 'UDI'], errors='ignore')
                    y_batch = batch_df['Machine failure']
                    perm_result = permutation_importance(
                        pipeline, X_batch, y_batch, scoring='f1', n_repeats=5, random_state=42
                    )
                    batch_importance_df = pd.DataFrame({
                        'Feature': X_batch.columns,
                        'Importance': perm_result.importances_mean
                    }).sort_values(by='Importance', ascending=False)

                    fig_batch = px.bar(
                        batch_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Batch Feature Importance Distribution',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig_batch.update_layout(
                        paper_bgcolor='#1e293b',
                        plot_bgcolor='#1e293b',
                        font=dict(color='#f8fafc'),
                        yaxis={'categoryorder': 'total ascending'},
                        height=350
                    )
                    st.plotly_chart(fig_batch, use_container_width=True)
                except Exception as err:
                    st.info(f"Batch feature importance calculation error: {err}")

        st.markdown("---")
        if results['drift_alert'] or st.button("TRIGGER TIER-2 AUTOMATED MODEL RETRAIN"):
            with st.spinner("Retraining master pipeline on updated telemetry logs..."):
                retrain_stats = execute_tier2_retrain(batch_df)
                st.success(f"Retraining Complete! Added {retrain_stats['new_records_added']} new records. Updated Validation F1: {retrain_stats['updated_f1']:.4f}")
                st.cache_resource.clear()

        st.markdown("---")
        st.subheader("Diagnostic Report Generation")
        
        batch_probs = pipeline.predict_proba(batch_df)[:, 1]
        active_thresh = tier2_threshold if results['active_tier'] == 'Tier 2 (Fallback)' else tier1_threshold
        batch_preds = (batch_probs >= active_thresh).astype(int)

        results_df = batch_df.copy()
        results_df['Failure_Probability'] = np.round(batch_probs, 4)
        results_df['Predicted_Status'] = np.where(batch_preds == 1, 'CRITICAL_FAULT', 'NOMINAL')

        with st.container(border=True):
            st.caption("BATCH INFERENCE SUMMARY")
            b_col1, b_col2, b_col3 = st.columns(3)
            total_machines = len(results_df)
            failures_flagged = int(np.sum(batch_preds))
            healthy_machines = total_machines - failures_flagged

            b_col1.metric("Total Equipment Logged", f"{total_machines:,}")
            b_col2.metric("Nominal Units", f"{healthy_machines:,}")
            b_col3.metric("Critical Faults Flagged", f"{failures_flagged:,}", delta_color="inverse")

        st.dataframe(results_df, use_container_width=True)

        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 DOWNLOAD DIAGNOSTIC REPORT CSV",
            data=csv_data,
            file_name="machinery_diagnostic_report.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# TAB 3: MODEL INTERPRETABILITY & GLOBAL SHAP
# -----------------------------------------------------------------------------
with tab3:
    st.header("Global Model Interpretability & Baseline Analytics")
    st.write("Inspect static baseline feature significance metrics generated during offline pipeline training.")

    if feature_importance_df is not None and not feature_importance_df.empty:
        with st.container(border=True):
            st.subheader("📊 Global Baseline Feature Importance")
            st.caption("Permutation importance evaluated on validation splits.")

            df_plot = feature_importance_df.copy()
            if 'Feature' in df_plot.columns:
                df_plot['Feature'] = df_plot['Feature'].apply(lambda x: str(x).split('__')[-1])

            fig_imp = px.bar(
                df_plot,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Global Feature Significance',
                color='Importance',
                color_continuous_scale='Tealgrn',
                labels={'Importance': 'Mean Score Drop', 'Feature': 'Feature Vector'}
            )
            fig_imp.update_layout(
                paper_bgcolor='#1e293b',
                plot_bgcolor='#1e293b',
                font=dict(color='#f8fafc'),
                yaxis={'categoryorder': 'total ascending'},
                height=450
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            with st.expander("📋 View Baseline Importance Raw Data"):
                st.dataframe(df_plot, use_container_width=True)

    else:
        st.info("💡 Feature importance metrics not detected in model payload.")

# -----------------------------------------------------------------------------
# TAB 4: MLOPS MODEL DECAY TIMELINE
# -----------------------------------------------------------------------------
with tab4:
    st.header("📈 Model Decay & Operational Audit Trail")
    st.write("Track live performance trends, metric degradation, and historical batch execution logs.")

    logs_df = fetch_historical_logs()

    if not logs_df.empty:
        fig_decay = px.line(
            logs_df,
            x='timestamp',
            y=['f1_score', 'precision_score', 'recall_score'],
            markers=True,
            title='Historical Model Performance Across Batch Inferences',
            labels={'value': 'Score', 'timestamp': 'Execution Time'}
        )
        fig_decay.update_layout(
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            font=dict(color='#f8fafc'),
            height=450
        )
        st.plotly_chart(fig_decay, use_container_width=True)

        st.subheader("Raw Operational Logs")
        st.dataframe(
            logs_df.sort_values(by='id', ascending=False), use_container_width=True
        )
    else:
        st.info("No execution logs found in SQLite database. Run batch inference to generate operational logs.")
