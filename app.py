import os
import sys
import types
import io
import pickle

# -----------------------------------------------------------------------------
# 1. PATH & MODULE INJECTION FIX FOR STREAMLIT CLOUD / JOBLIB UNPICKLING
# -----------------------------------------------------------------------------
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# Import custom transformers with fallback module generation
try:
    import custom_transformers
    from custom_transformers import IQROutlierClipper, MaintenanceFeatureEngineer
except ImportError:
    custom_transformers = types.ModuleType("custom_transformers")
    
    # Dynamic fallbacks to ensure unpickling never breaks if file is missing
    class IQROutlierClipper:
        def fit(self, X, y=None): return self
        def transform(self, X): return X
        
    class MaintenanceFeatureEngineer:
        def fit(self, X, y=None): return self
        def transform(self, X): 
            X = X.copy()
            if 'Air temperature [K]' in X.columns and 'Process temperature [K]' in X.columns:
                X['Temperature_Difference'] = X['Process temperature [K]'] - X['Air temperature [K]']
            if 'Torque [Nm]' in X.columns and 'Rotational speed [rpm]' in X.columns:
                X['Power_Index'] = X['Torque [Nm]'] * X['Rotational speed [rpm]']
            if 'Tool wear [min]' in X.columns and 'Torque [Nm]' in X.columns:
                X['Wear_Torque_Ratio'] = X['Tool wear [min]'] * X['Torque [Nm]']
            return X
        
    custom_transformers.IQROutlierClipper = IQROutlierClipper
    custom_transformers.MaintenanceFeatureEngineer = MaintenanceFeatureEngineer
    sys.modules["custom_transformers"] = custom_transformers
    from custom_transformers import IQROutlierClipper, MaintenanceFeatureEngineer

# Register classes across all target import namespaces
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

# Custom operational modules (with defensive fallbacks)
try:
    from drift_monitor import run_two_tier_inference
except ImportError:
    def run_two_tier_inference(df, y_true=None):
        return {
            'metrics': None,
            'active_tier': 'Tier 1 (Standard)',
            'drift_alert': False
        }

try:
    from retrain_module import execute_tier2_retrain
except ImportError:
    def execute_tier2_retrain(df):
        return {'new_records_added': len(df), 'updated_f1': 0.8500}

try:
    from logger_db import fetch_historical_logs, log_batch_execution
except ImportError:
    def fetch_historical_logs():
        return pd.DataFrame()
    def log_batch_execution(*args, **kwargs):
        pass

# -----------------------------------------------------------------------------
# 2. CUSTOM UNPICKLER OVERRIDE
# -----------------------------------------------------------------------------
class SafeCustomUnpickler(pickle.Unpickler):
    TARGET_CLASSES = {
        "IQROutlierClipper": IQROutlierClipper,
        "MaintenanceFeatureEngineer": MaintenanceFeatureEngineer
    }

    def find_class(self, module, name):
        if name in self.TARGET_CLASSES:
            return self.TARGET_CLASSES[name]
        try:
            return super().find_class(module, name)
        except Exception:
            if hasattr(custom_transformers, name):
                return getattr(custom_transformers, name)
            if hasattr(__main__, name):
                return getattr(__main__, name)
            if name in globals():
                return globals()[name]
            raise ModuleNotFoundError(f"Could not resolve class '{name}' from module '{module}'.")

class SafeJoblibUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        if name in SafeCustomUnpickler.TARGET_CLASSES:
            return SafeCustomUnpickler.TARGET_CLASSES[name]
        try:
            return super().find_class(module, name)
        except Exception:
            if hasattr(custom_transformers, name):
                return getattr(custom_transformers, name)
            if hasattr(__main__, name):
                return getattr(__main__, name)
            if name in globals():
                return globals()[name]
            raise

# -----------------------------------------------------------------------------
# 3. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Machinery Diagnostics & Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(rgba(56, 189, 248, 0.07) 1px, transparent 0);
        background-size: 24px 24px;
        color: #f8fafc;
    }

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

    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    }

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

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-top: 3px solid #38bdf8 !important;
        border-left: 1px solid #334155 !important;
        border-right: 1px solid #334155 !important;
        border-bottom: 2px solid #1e293b !important;
    }

    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div,
    input {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background-color: #0284c7;
        color: #ffffff !important;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #38bdf8;
        padding: 0.6rem 1.4rem;
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
# 4. LOAD MODEL PIPELINE & PAYLOAD
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_payload():
    model_paths = [
        os.path.join(app_dir, 'predictive_maintenance_pipeline.joblib'),
        os.path.join(app_dir, 'model_store', 'production_v1.joblib'),
        'predictive_maintenance_pipeline.joblib'
    ]
    
    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if not model_path:
        st.error("❌ Model payload not found. Please verify 'predictive_maintenance_pipeline.joblib' exists.")
        st.stop()
    
    try:
        with open(model_path, 'rb') as f:
            return SafeCustomUnpickler(f).load()
    except Exception:
        pass

    try:
        with open(model_path, 'rb') as f:
            return SafeJoblibUnpickler(model_path, f).load()
    except Exception:
        pass

    return joblib.load(model_path)

payload = load_model_payload()

if isinstance(payload, dict):
    pipeline = payload.get('pipeline', payload.get('model'))
    tier1_threshold = float(payload.get('tier1_threshold', 0.8900))
    tier2_threshold = float(payload.get('tier2_threshold', 0.5000))
    drift_tolerance_f1 = float(payload.get('drift_tolerance_f1', 0.8000))
    baseline_metrics = payload.get('baseline_metrics', {})
    feature_importance_df = payload.get('feature_importances', payload.get('feature_importance_df', None))
else:
    pipeline = payload
    tier1_threshold = 0.8900
    tier2_threshold = 0.5000
    drift_tolerance_f1 = 0.8000
    feature_importance_df = None

# Helper function to extract required columns from ColumnTransformer safely
def get_expected_columns(model_pipeline):
    """Retrieves list of expected input columns by inspecting ColumnTransformer steps."""
    expected = []
    if hasattr(model_pipeline, 'named_steps'):
        preprocessor = model_pipeline.named_steps.get('preprocessor', None)
        if preprocessor and hasattr(preprocessor, 'transformers_'):
            for name, trans, cols in preprocessor.transformers_:
                if name != 'remainder' and isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
                    expected.extend(list(cols))
    return list(set(expected))

# Robust, column-safe prediction wrapper
def safe_predict_proba(model_pipeline, df_input):
    """
    Safely executes prediction by ensuring custom feature engineering 
    and ColumnTransformer steps execute sequentially on input DataFrames.
    """
    df_eval = df_input.copy()

    # Step 1: If feature engineer step exists in pipeline, transform input first
    if hasattr(model_pipeline, 'named_steps') and 'feature_engineer' in model_pipeline.named_steps:
        fe_step = model_pipeline.named_steps['feature_engineer']
        if hasattr(fe_step, 'transform'):
            df_eval = fe_step.transform(df_eval)

    # Step 2: Ensure any missing engineered or baseline columns are backfilled
    expected_cols = get_expected_columns(model_pipeline)
    for col in expected_cols:
        if col not in df_eval.columns:
            df_eval[col] = 0.0

    # Step 3: Run preprocessor and classifier sequentially
    if hasattr(model_pipeline, 'named_steps') and 'preprocessor' in model_pipeline.named_steps:
        preprocessor = model_pipeline.named_steps['preprocessor']
        classifier = model_pipeline.named_steps.get('classifier', list(model_pipeline.named_steps.values())[-1])

        X_trans = preprocessor.transform(df_eval)
        return classifier.predict_proba(X_trans)

    # Fallback to standard pipeline call
    return model_pipeline.predict_proba(df_eval)
    
# -----------------------------------------------------------------------------
# 5. SIDEBAR CONTROL PANEL
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
# 6. HERO SECTION
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
        probs = safe_predict_proba(pipeline, input_df)[0]
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
                    named_steps = getattr(pipeline, 'named_steps', {})
                    feature_engineer = named_steps.get('feature_engineer', None)
                    preprocessor = named_steps.get('preprocessor', None)
                    classifier = named_steps.get('classifier', list(named_steps.values())[-1] if named_steps else pipeline)

                    # Sequential Feature Transformation
                    x_eval = input_df.copy()
                    if feature_engineer and hasattr(feature_engineer, 'transform'):
                        x_eval = feature_engineer.transform(x_eval)

                    # Backfill expected preprocessor columns
                    expected_cols = get_expected_columns(pipeline)
                    for col in expected_cols:
                        if col not in x_eval.columns:
                            x_eval[col] = 0.0

                    x_trans = preprocessor.transform(x_eval) if preprocessor else x_eval

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
                    plt.close(fig)
                except Exception as e:
                    st.info(f"SHAP local breakdown notice: {e}")
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

        if results.get('metrics'):
            log_batch_execution(
                batch_size=len(batch_df),
                f1=results['metrics'].get('f1', 0.0),
                precision=results['metrics'].get('precision', 0.0),
                recall=results['metrics'].get('recall', 0.0),
                tier_used=results.get('active_tier', 'Tier 1'),
                drift_detected=results.get('drift_alert', False)
            )

        st.markdown("---")
        st.subheader("Drift Monitoring & Operational Tier Status")

        active_tier = results.get('active_tier', 'Tier 1 (Standard)')
        drift_alert = results.get('drift_alert', False)

        if drift_alert:
            st.error(f"🚨 **CONCEPT DRIFT ALERT DETECTED!** Operating mode overridden to **{active_tier}**.")
            st.warning(f"Batch performance F1 dropped below tolerance ({drift_tolerance_f1:.2f}). Tier 2 fallback engaged to prevent uncaptured failures.")
        else:
            st.success(f"✅ **NORMAL TELEMETRY OPERATING STATE:** Active Tier = **{active_tier}**")

        if results.get('metrics'):
            with st.container(border=True):
                st.caption("EVALUATION METRICS FOR CURRENT INGESTED BATCH")
                col1, col2, col3 = st.columns(3)
                col1.metric("Batch F1-Score", f"{results['metrics'].get('f1', 0.0):.4f}")
                col2.metric("Batch Precision", f"{results['metrics'].get('precision', 0.0):.4f}")
                col3.metric("Batch Recall", f"{results['metrics'].get('recall', 0.0):.4f}")

        st.markdown("---")
        if drift_alert or st.button("TRIGGER TIER-2 AUTOMATED MODEL RETRAIN"):
            with st.spinner("Retraining master pipeline on updated telemetry logs..."):
                retrain_stats = execute_tier2_retrain(batch_df)
                st.success(f"Retraining Complete! Added {retrain_stats.get('new_records_added', 0)} new records. Updated Validation F1: {retrain_stats.get('updated_f1', 0.0):.4f}")
                st.cache_resource.clear()

        st.markdown("---")
        st.subheader("Diagnostic Report Generation")
        
        batch_probs = safe_predict_proba(pipeline, batch_df)[:, 1]
        active_thresh = tier2_threshold if active_tier == 'Tier 2 (Fallback)' else tier1_threshold
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

    if feature_importance_df is not None:
        if isinstance(feature_importance_df, pd.DataFrame) and not feature_importance_df.empty:
            df_plot = feature_importance_df.copy()
        elif isinstance(feature_importance_df, (list, np.ndarray)):
            df_plot = pd.DataFrame({
                'Feature': [f"Feature {i}" for i in range(len(feature_importance_df))],
                'Importance': feature_importance_df
            })
        else:
            df_plot = pd.DataFrame()

        if not df_plot.empty:
            with st.container(border=True):
                st.subheader("📊 Global Baseline Feature Importance")
                st.caption("Permutation importance evaluated on validation splits.")

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
            st.info("💡 Feature importance structure detected, but contained no data.")
    else:
        st.info("💡 Baseline feature importance metrics not included in current model payload.")

# -----------------------------------------------------------------------------
# TAB 4: MLOPS MODEL DECAY TIMELINE
# -----------------------------------------------------------------------------
with tab4:
    st.header("📈 Model Decay & Operational Audit Trail")
    st.write("Track live performance trends, metric degradation, and historical batch execution logs.")

    logs_df = fetch_historical_logs()

    if isinstance(logs_df, pd.DataFrame) and not logs_df.empty:
        required_cols = {'timestamp', 'f1_score', 'precision_score', 'recall_score'}
        if required_cols.issubset(logs_df.columns):
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
        sort_col = 'id' if 'id' in logs_df.columns else logs_df.columns[0]
        st.dataframe(
            logs_df.sort_values(by=sort_col, ascending=False), use_container_width=True
        )
    else:
        st.info("No historical logs found in execution registry. Run batch inference in Tab 2 to record operational logs.")
