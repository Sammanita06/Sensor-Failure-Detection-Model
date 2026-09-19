import os
import joblib
import pandas as pd
from custom_transformers import MaintenanceFeatureEngineer, IQROutlierClipper

def test_custom_transformers():
    engineer = MaintenanceFeatureEngineer()
    df = pd.DataFrame({
        'Air temperature [K]': [300.0],
        'Process temperature [K]': [310.0],
        'Rotational speed [rpm]': [1500],
        'Torque [Nm]': [40.0],
        'Tool wear [min]': [10],
        'Type': ['M']
    })
    transformed = engineer.transform(df)
    assert 'Temperature_Difference' in transformed.columns or 'Temp_Ratio' in transformed.columns, "Feature engineering transformation failed."
    print("✅ Custom transformers working as expected.")

def test_pipeline_loading():
    model_path = 'predictive_maintenance_pipeline.joblib'
    if os.path.exists(model_path):
        payload = joblib.load(model_path)
        assert payload is not None, "Pipeline payload loaded as None."
        print("✅ Pipeline payload loaded successfully.")
    else:
        print("⚠️ Model file 'predictive_maintenance_pipeline.joblib' not found. Skipping file existence check for CI.")

if __name__ == "__main__":
    test_custom_transformers()
    test_pipeline_loading()
