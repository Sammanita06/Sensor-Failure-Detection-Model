import joblib
import pandas as pd
from custom_transformers import MaintenanceFeatureEngineer, IQROutlierClipper

def test_pipeline_loading():
    # Load model payload
    payload = joblib.load('predictive_maintenance_pipeline.joblib')
    pipeline = payload['pipeline']
    assert pipeline is not None, "Pipeline failed to load."
    print("✅ Model payload loaded successfully.")

def test_custom_transformers():
    # Verify custom feature engineer
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
    assert 'Temperature_Difference' in transformed.columns, "Feature engineering failed."
    print("✅ Custom transformers working as expected.")

if __name__ == "__main__":
    test_pipeline_loading()
    test_custom_transformers()
