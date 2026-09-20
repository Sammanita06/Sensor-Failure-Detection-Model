import os
import sys
import io
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import sklearn
import __main__

# -----------------------------------------------------------------------------
# 1. PATH RESOLUTION & MODULE BINDINGS
# -----------------------------------------------------------------------------
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Import custom transformers directly from custom_transformers.py
try:
    import custom_transformers
    from custom_transformers import IQROutlierClipper, MaintenanceFeatureEngineer
    
    # Register directly in sys.modules and __main__ for unpickler visibility
    sys.modules['custom_transformers'] = custom_transformers
    setattr(__main__, 'IQROutlierClipper', IQROutlierClipper)
    setattr(__main__, 'MaintenanceFeatureEngineer', MaintenanceFeatureEngineer)
except ImportError as e:
    st.error(f"❌ Failed to load custom_transformers.py: {e}")
    st.stop()

# Import auxiliary modules with fallback definitions
try:
    import drift_monitor
except ImportError:
    class drift_monitor:
        @staticmethod
        def calculate_ks_drift(ref_df, curr_df):
            return {"drift_detected": False, "p_value": 1.0, "stat": 0.0}

try:
    import retrain_module
except ImportError:
    class retrain_module:
        @staticmethod
        def execute_tier2_retrain(new_batch_df):
            return {"status": "retrain_module.py not available"}

try:
    import logger_db
except ImportError:
    class logger_db:
        @staticmethod
        def log_batch_execution(batch_size, f1, precision, recall, tier_used, drift_detected):
            pass
        @staticmethod
        def fetch_historical_logs():
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sensor Failure Detection & MLOps System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 3. SAFE UNPICKLER DEFINITION FOR PYTHON 3.14 + JOBLIB
# -----------------------------------------------------------------------------
class SafeCustomUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    """Overrides Joblib unpickling to force class lookup from custom_transformers.py."""
    
    TARGET_CLASSES = {
        "IQROutlierClipper": IQROutlierClipper,
        "MaintenanceFeatureEngineer": MaintenanceFeatureEngineer
    }

    def find_class(self, module, name):
        # 1. Direct local module override
        if name in self.TARGET_CLASSES:
            return self.TARGET_CLASSES[name]
        if hasattr(custom_transformers, name):
            return getattr(custom_transformers, name)
            
        # 2. Standard resolution fallback
        try:
            return super().find_class(module, name)
        except Exception:
            if hasattr(__main__, name):
                return getattr(__main__, name)
            if name in globals():
                return globals()[name]
            raise ModuleNotFoundError(f"Could not resolve class '{name}' from module '{module}'.")

# -----------------------------------------------------------------------------
# 4. LOAD MODEL PIPELINE & PAYLOAD
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_payload():
    model_path = os.path.join(app_dir, 'predictive_maintenance_pipeline.joblib')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(app_dir, 'model_store', 'production_v1.joblib')
        
    if not os.path.exists(model_path):
        st.error("❌ Model payload not found. Please place 'predictive_maintenance_pipeline.joblib' in the repository root or 'model_store/'.")
        st.stop()

    try:
        with open(model_path, 'rb') as f:
            unpickler = SafeCustomUnpickler(model_path, f)
            return unpickler.load()
    except Exception:
        import joblib.numpy_pickle
        original_unpickler = joblib.numpy_pickle.NumpyUnpickler
        try:
            joblib.numpy_pickle.NumpyUnpickler = SafeCustomUnpickler
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"❌ Failed to unpickle model file: {e}")
            st.stop()
        finally:
            joblib.numpy_pickle.NumpyUnpickler = original_unpickler

# Load payload
payload = load_model_payload()

if isinstance(payload, dict):
    model_pipeline = payload.get('pipeline', payload.get('model'))
    training_metadata = payload.get('baseline_metrics', {})
    optimal_threshold = payload.get('optimal_threshold', 0.50)
else:
    model_pipeline = payload
    training_metadata = {}
    optimal_threshold = 0.50

# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Navigation & Status")
st.sidebar.success("✅ Model Pipeline Loaded Successfully")

if training_metadata:
    with st.sidebar.expander("📌 Baseline Model Metrics", expanded=False):
        st.write(f"**Baseline F1 Score:** {training_metadata.get('f1', 'N/A')}")
        st.write(f"**Precision:** {training_metadata.get('precision', 'N/A')}")
        st.write(f"**Recall:** {training_metadata.get('recall', 'N/A')}")
        st.write(f"**Operating Threshold:** {optimal_threshold:.2f}")

nav_choice = st.sidebar.radio(
    "Select Interface",
    ["Predictive Diagnostics", "Batch Inference", "Data Drift Monitor", "Retraining Control"]
)

# -----------------------------------------------------------------------------
# 6. TAB 1: PREDICTIVE DIAGNOSTICS (SINGLE INFERENCE)
# -----------------------------------------------------------------------------
if nav_choice == "Predictive Diagnostics":
    st.title("⚙️ Real-Time Sensor Failure Diagnostics")
    st.markdown("Adjust input operational values to compute failure predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        type_input = st.selectbox("Type", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=350.0, value=300.0)
        process_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=350.0, value=310.0)
        
    with col2:
        rotational_speed = st.number_input("Rotational speed [rpm]", min_value=500, max_value=3000, value=1500)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
        tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=300, value=0)

    input_data = pd.DataFrame([{
        'Type': type_input,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }])

    if st.button("🔍 Run Failure Analysis"):
        try:
            probs = model_pipeline.predict_proba(input_data)[:, 1] if hasattr(model_pipeline, "predict_proba") else [0.0]
            probability = float(probs[0])
            prediction = 1 if probability >= optimal_threshold else 0
            
            st.subheader("Diagnostic Results")
            if prediction == 1:
                st.error(f"⚠️ **FAILURE RISK DETECTED** (Probability: {probability:.2%})")
            else:
                st.success(f"✅ **NORMAL OPERATION** (Failure Probability: {probability:.2%})")
        except Exception as e:
            st.error(f"Inference execution failed: {e}")

# -----------------------------------------------------------------------------
# 7. TAB 2: BATCH INFERENCE
# -----------------------------------------------------------------------------
elif nav_choice == "Batch Inference":
    st.title("📂 Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV File for Batch Inference", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", batch_df.head())
        
        if st.button("Run Batch Predictions"):
            try:
                if hasattr(model_pipeline, "predict_proba"):
                    probs = model_pipeline.predict_proba(batch_df)[:, 1]
                    preds = (probs >= optimal_threshold).astype(int)
                    batch_df['Failure_Probability'] = probs
                else:
                    preds = model_pipeline.predict(batch_df)
                
                batch_df['Failure_Prediction'] = preds
                
                st.success("Batch Inference Complete!")
                st.dataframe(batch_df)
                
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error processing batch: {e}")

# -----------------------------------------------------------------------------
# 8. TAB 3: DATA DRIFT MONITOR
# -----------------------------------------------------------------------------
elif nav_choice == "Data Drift Monitor":
    st.title("📊 Data Drift & System Health")
    st.info("Monitors statistical features and SQLite operation logs.")
    
    try:
        logs_df = logger_db.fetch_historical_logs()
        if not logs_df.empty:
            st.subheader("Operational Execution Logs")
            st.dataframe(logs_df)
        else:
            st.warning("No historical logs recorded in the database yet.")
    except Exception as e:
        st.error(f"Error loading logs: {e}")

# -----------------------------------------------------------------------------
# 9. TAB 4: RETRAINING CONTROL
# -----------------------------------------------------------------------------
elif nav_choice == "Retraining Control":
    st.title("🔄 MLOps Model Retraining")
    st.warning("Appends new batch telemetry into master data storage and retrains pipeline artifacts.")
    
    retrain_file = st.file_uploader("Upload New Telemetry Data for Retraining", type=["csv"])
    if retrain_file is not None:
        new_batch_df = pd.read_csv(retrain_file)
        if st.button("Trigger Retrain Task"):
            try:
                result = retrain_module.execute_tier2_retrain(new_batch_df)
                st.success("Retraining execution successful!")
                st.json(result)
            except Exception as e:
                st.error(f"Retraining task failed: {e}")
