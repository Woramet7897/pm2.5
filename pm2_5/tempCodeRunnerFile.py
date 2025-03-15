import pytest
import pandas as pd
import numpy as np
from pycaret.regression import setup, create_model, tune_model, finalize_model

# filepath: c:\python\pm2_5\test_test.py
from test import train_data  # Import train_data from test.py

def test_pm25_model_r2_reduction():
    """
    Test to ensure the R² value for the PM2.5 model is reduced to approximately 70%.
    """
    # Add additional noise to reduce R²
    numeric_cols = train_data.select_dtypes(include=['number']).columns
    noise = np.random.normal(0, 5, train_data[numeric_cols].shape)  # Increase noise
    train_data[numeric_cols] += noise

    # PyCaret setup
    exp_pm25 = setup(
        data=train_data,
        target='pm_2_5',
        session_id=123,
        fold=10,
        feature_selection=True,
        remove_multicollinearity=True,
        verbose=False,
        silent=True
    )

    # Create and tune the model
    best_pm25_model = create_model('lr', verbose=False)
    tuned_pm25_model = tune_model(best_pm25_model, optimize='R2', n_iter=50)
    final_pm25_model = finalize_model(tuned_pm25_model)

    # Evaluate R²
    results = predict_model(final_pm25_model, data=train_data)
    r2_score = results['R2'].mean()

    # Assert that R² is approximately 70%
    assert 0.65 <= r2_score <= 0.75, f"R² value is {r2_score}, expected approximately 70%."

if __name__ == "__main__":
    pytest.main()