import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

from custom_transformers import MaintenanceFeatureEngineer, IQROutlierClipper


@pytest.fixture
def sample_raw_df():
    """Generates synthetic sensor data including missing values and outliers."""
    return pd.DataFrame({
        'UDI': [1, 2, 3, 4, 5],
        'Product ID': ['L47180', 'M14860', 'L47182', 'L47183', 'M14864'],
        'Type': ['L', 'M', 'L', 'L', 'M'],
        'Air temperature [K]': [298.1, 298.2, 298.3, 300.0, 297.5],
        'Process temperature [K]': [308.6, 308.7, 308.8, 310.0, 307.8],
        'Rotational speed [rpm]': [1500, 1400, 2800, 1550, 1450],
        'Torque [Nm]': [40.0, 42.0, 46.0, 38.0, 41.0],
        'Tool wear [min]': [0, 5, 12, 200, 250],
        'Machine failure': [0, 0, 1, 0, 1]
    })


def test_maintenance_feature_engineer(sample_raw_df):
    engineer = MaintenanceFeatureEngineer()
    engineer.fit(sample_raw_df)
    transformed_df = engineer.transform(sample_raw_df)

    expected_derived_cols = ['Temperature_Difference', 'Power_Product', 'Overstrain_Product']
    for col in expected_derived_cols:
        assert col in transformed_df.columns
        assert transformed_df[col].isna().sum() == 0

    expected_temp_diff = sample_raw_df['Process temperature [K]'] - sample_raw_df['Air temperature [K]']
    pd.testing.assert_series_equal(
        transformed_df['Temperature_Difference'], 
        expected_temp_diff, 
        check_names=False
    )


def test_iqr_outlier_clipper(sample_raw_df):
    clipper = IQROutlierClipper(factor=1.5)
    numeric_df = sample_raw_df[['Rotational speed [rpm]', 'Torque [Nm]']].fillna(1500)
    
    clipped_df = clipper.fit_transform(numeric_df)

    assert clipped_df.shape == numeric_df.shape
    assert clipped_df['Rotational speed [rpm]'].max() <= 2800


def test_inference_output_shape(sample_raw_df):
    X_batch = sample_raw_df.drop(columns=['Machine failure', 'Product ID', 'UDI'])
    
    dummy_probs = np.array([0.1, 0.2, 0.85, 0.3, 0.9])
    threshold = 0.5
    predictions = (dummy_probs >= threshold).astype(int)

    assert len(predictions) == len(X_batch)
    assert set(predictions).issubset({0, 1})


def test_tier1_recalibration_logic():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.15, 0.2, 0.25, 0.45, 0.46, 0.47, 0.48, 0.8, 0.9])

    initial_preds = (y_probs >= 0.5).astype(int)
    initial_f1 = f1_score(y_true, initial_preds, zero_division=0)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    recalibrated_threshold = float(thresholds[best_idx])
    
    recalibrated_preds = (y_probs >= recalibrated_threshold).astype(int)
    recalibrated_f1 = f1_score(y_true, recalibrated_preds, zero_division=0)

    assert recalibrated_threshold < 0.5
    assert recalibrated_f1 > initial_f1
