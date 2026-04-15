"""
train_model.py — Full training pipeline for the Real Estate Value Estimator.

Usage:
    python model/train_model.py

Reads   : data/train.csv  (Ames Housing Dataset from Kaggle)
Outputs : model/xgb_model.pkl, model/feature_names.pkl
"""

import os
import sys

# Ensure project root is on the path so `utils` can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.preprocessing import (
    load_and_clean,
    engineer_features,
    impute_missing,
    encode_features,
)


def train_and_save():
    """Run the full train → evaluate → save pipeline."""

    data_path = os.path.join(PROJECT_ROOT, "data", "AmesHousing.csv")
    if not os.path.isfile(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please place the AmesHousing.csv dataset in the /data folder.")
        print("URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data")
        sys.exit(1)

    print("=" * 60)
    print("  Real Estate Value Estimator — Model Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Preprocessing pipeline
    # ------------------------------------------------------------------
    print("\n[1/5] Loading and cleaning data ...")
    df = load_and_clean(data_path)
    print(f"      Rows after cleaning: {len(df)}")

    print("[2/5] Engineering features ...")
    df = engineer_features(df)

    print("[3/5] Imputing missing values ...")
    df = impute_missing(df)

    print("[4/5] Encoding categorical features ...")
    df = encode_features(df)

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Training samples : {len(X_train)}")
    print(f"      Test samples     : {len(X_test)}")
    print(f"      Feature count    : {X.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Train XGBoost
    # ------------------------------------------------------------------
    print("\n[5/5] Training XGBoost model ...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    preds_log = model.predict(X_test)
    rmse_log = np.sqrt(mean_squared_error(y_test, preds_log))

    # Also evaluate in real-dollar space
    preds_dollar = np.expm1(preds_log)
    actual_dollar = np.expm1(y_test)
    rmse_dollar = np.sqrt(mean_squared_error(actual_dollar, preds_dollar))
    mae_dollar = mean_absolute_error(actual_dollar, preds_dollar)
    r2 = r2_score(actual_dollar, preds_dollar)

    print("\n" + "-" * 40)
    print("  Model Performance")
    print("-" * 40)
    print(f"  RMSE (log scale)  : {rmse_log:.4f}")
    print(f"  RMSE (dollars)    : ${rmse_dollar:,.0f}")
    print(f"  MAE  (dollars)    : ${mae_dollar:,.0f}")
    print(f"  R² Score          : {r2:.4f}")
    print("-" * 40)

    # ------------------------------------------------------------------
    # 5. Save artifacts
    # ------------------------------------------------------------------
    model_dir = os.path.join(PROJECT_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgb_model.pkl")
    features_path = os.path.join(model_dir, "feature_names.pkl")

    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), features_path)

    print(f"\n✅ Model saved    → {model_path}")
    print(f"✅ Features saved → {features_path}")
    print(f"\nTotal features: {len(X.columns)}")
    print("\nYou can now run:  streamlit run app.py")


if __name__ == "__main__":
    train_and_save()
