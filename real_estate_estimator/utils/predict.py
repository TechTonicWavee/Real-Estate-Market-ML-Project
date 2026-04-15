"""
predict.py — Prediction logic using the saved XGBoost model.

Loads the trained model + feature list once, then aligns any new input
to the exact training schema before predicting.
"""

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import engineer_features, impute_missing, encode_features


def load_model(model_path: str = "model/xgb_model.pkl",
               features_path: str = "model/feature_names.pkl"):
    """Load the persisted model and feature-name list."""
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return model, feature_names


def predict_price(input_dict: dict, model, feature_names: list):
    """
    Predict house price from a raw feature dictionary.

    Parameters
    ----------
    input_dict : dict
        Raw feature values coming from the Streamlit UI form.
    model : trained XGBRegressor
        The loaded model object.
    feature_names : list[str]
        Column names the model was trained on.

    Returns
    -------
    actual_price : float
        Predicted price in real dollars (reverse of log1p transform).
    aligned_df : pd.DataFrame
        Single-row DataFrame aligned to training features (for SHAP).
    """
    df = pd.DataFrame([input_dict])

    # Run the same transformations used during training
    df = engineer_features(df)
    df = impute_missing(df)
    df = encode_features(df)

    # Align columns to training schema
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Predict (model outputs log1p price)
    log_price = model.predict(df)[0]
    actual_price = float(np.expm1(log_price))  # reverse log1p

    return round(actual_price, 2), df
