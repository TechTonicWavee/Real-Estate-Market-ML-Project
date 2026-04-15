"""
preprocessing.py — Full data cleaning & feature engineering pipeline
for the Ames Housing Dataset (80+ features).

Handles:
  • Loading & cleaning raw CSV
  • Feature engineering (domain-informed composite features)
  • Missing-value imputation (category-aware)
  • Encoding (one-hot for nominal columns)
"""

import datetime
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1A. Load & Clean
# ---------------------------------------------------------------------------
def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load the Ames Housing CSV, drop ID, remove outliers, log-transform target."""
    df = pd.read_csv(filepath)

    # Dynamically fix column names (strip spaces, fix specific names) so both Kaggle format 
    # and raw AmesHousing.csv formats work without crashing the pipeline.
    df.columns = [str(c).replace(" ", "") for c in df.columns]
    df = df.rename(columns={"YearRemod/Add": "YearRemodAdd", "PID": "Id"})

    # Drop ID column (not a feature)
    df.drop("Id", axis=1, inplace=True, errors="ignore")
    df.drop("Order", axis=1, inplace=True, errors="ignore")

    # Remove luxury outliers — houses > 4000 sqft skew average-home predictions
    if "GrLivArea" in df.columns:
        df = df[df["GrLivArea"] < 4000]

    # Log-transform target (SalePrice is heavily right-skewed)
    if "SalePrice" in df.columns:
        df["SalePrice"] = np.log1p(df["SalePrice"])

    return df


# ---------------------------------------------------------------------------
# 1B. Feature Engineering — domain knowledge encoded as features
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create composite features that capture real-estate domain knowledge."""
    current_year = datetime.datetime.now().year

    # --- Age features ---
    if "YearBuilt" in df.columns:
        df["HouseAge"] = current_year - df["YearBuilt"]
    if "YearRemodAdd" in df.columns:
        df["YearsSinceRemodel"] = current_year - df["YearRemodAdd"]
    if "GarageYrBlt" in df.columns:
        df["GarageAge"] = current_year - df["GarageYrBlt"].fillna(
            df.get("YearBuilt", current_year)
        )

    # --- Area combinations ---
    area_cols = ["BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF"]
    if all(c in df.columns for c in area_cols):
        df["TotalSF"] = df[area_cols].sum(axis=1)

    bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    if all(c in df.columns for c in bath_cols):
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    if all(c in df.columns for c in porch_cols):
        df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

    # --- Luxury / Quality composite score ---
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0, "None": 0}
    garage_finish_map = {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0, "None": 0}

    luxury = pd.Series(0, index=df.index, dtype=float)
    if "OverallQual" in df.columns:
        luxury += df["OverallQual"] * 2
    if "KitchenQual" in df.columns:
        luxury += df["KitchenQual"].map(quality_map).fillna(0)
    if "FireplaceQu" in df.columns:
        luxury += df["FireplaceQu"].map(quality_map).fillna(0)
    if "GarageFinish" in df.columns:
        luxury += df["GarageFinish"].map(garage_finish_map).fillna(0)
    df["OverallLuxury"] = luxury

    # --- Boolean flags ---
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    if "HouseStyle" in df.columns:
        df["Is2Story"] = df["HouseStyle"].astype(str).str.contains("2").astype(int)

    return df


# ---------------------------------------------------------------------------
# 1C. Handle Missing Values
# ---------------------------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with domain-appropriate strategies."""

    # Categorical cols where NA means "feature doesn't exist"
    cat_none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType",
    ]
    for col in cat_none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numeric cols where NA means "feature doesn't exist" → fill with 0
    num_zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
    ]
    for col in num_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Remaining numeric: median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Remaining categorical: mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "None")

    return df


# ---------------------------------------------------------------------------
# 1D. Encoding
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all remaining categorical (object) columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------
def full_pipeline(filepath: str) -> pd.DataFrame:
    """Run the entire preprocessing pipeline end-to-end."""
    df = load_and_clean(filepath)
    df = engineer_features(df)
    df = impute_missing(df)
    df = encode_features(df)
    return df
