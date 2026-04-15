"""
app.py — Main Streamlit entry point for the Real Estate Value Estimator.

Run with:
    streamlit run app.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Value Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.isfile(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    # Fallback inline CSS (essential bits)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .main { background-color: #f8f9fb; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 14px 32px !important; font-size: 1rem !important; font-weight: 600 !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_cached():
    from utils.predict import load_model
    return load_model()


@st.cache_data
def load_training_data():
    """Load the raw training data for market-context statistics."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    if os.path.isfile(data_path):
        df = pd.read_csv(data_path)
        df.drop("Id", axis=1, inplace=True, errors="ignore")
        df = df[df["GrLivArea"] < 4000]
        return df
    return None


# ---------------------------------------------------------------------------
# Check that model files exist
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "model", "feature_names.pkl")
MODEL_READY = os.path.isfile(MODEL_PATH) and os.path.isfile(FEATURES_PATH)


# ---------------------------------------------------------------------------
# Constants — dropdown options (based on Ames Housing Dataset values)
# ---------------------------------------------------------------------------
NEIGHBORHOODS = [
    "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
    "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW",
    "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber",
    "IDOTRR", "ClearCr", "StoneBr", "SWISU", "Blmngtn",
    "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste",
]

HOUSE_STYLES = ["1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"]

ROOF_MATERIALS = ["CompShg", "Tar&Grv", "WdShake", "WdShngl", "Metal", "Membran", "Roll"]

FOUNDATION_TYPES = ["PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"]

KITCHEN_QUALITY_MAP = {"Excellent": "Ex", "Good": "Gd", "Average": "TA", "Fair": "Fa"}

EXTERIOR_OPTIONS = [
    "VinylSd", "HdBoard", "MetalSd", "Wd Sdng", "Plywood",
    "CemntBd", "BrkFace", "WdShing", "Stucco", "AsbShng",
]

SALE_CONDITION = ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"]

SALE_TYPE = ["WD", "New", "COD", "ConLD", "ConLI", "ConLw", "CWD", "Oth", "Con"]


# ═══════════════════════════════════════════════════════════════════════════
# APP HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="app-title">🏠 Real Estate Value Estimator</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">AI-powered property valuation with full explainability · Powered by XGBoost & SHAP</p>',
    unsafe_allow_html=True,
)

if not MODEL_READY:
    st.warning(
        "⚠️ **Model not trained yet.**  \n"
        "1. Place `train.csv` in the `/data` folder  \n"
        "2. Run `python model/train_model.py`  \n"
        "3. Refresh this page."
    )

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, = st.tabs(["🏡 Estimate Your Home"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    left_col, spacer, right_col = st.columns([1.2, 0.1, 1.8])

    # ------------------------------------------------------------------
    # LEFT COLUMN — Input Panel
    # ------------------------------------------------------------------
    with left_col:
        st.markdown('<div class="input-panel-header">📋 Basic Details</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=0)
        with col_b:
            house_style = st.selectbox("House Style", HOUSE_STYLES, index=0)

        overall_qual = st.slider("Overall Quality", 1, 10, 7, help="Rate the overall material and finish of the house")

        # ---- Size & Space ----
        st.markdown('<div class="input-panel-header">📐 Size & Space</div>', unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 300, 3999, 1500, step=50)
            total_bsmt_sf = st.number_input("Total Basement Area (sqft)", 0, 3000, 800, step=50)
        with col_d:
            bedrooms = st.slider("Bedrooms Above Ground", 0, 8, 3)
            full_bath = st.slider("Full Bathrooms", 0, 4, 2)

        col_e, col_f = st.columns(2)
        with col_e:
            half_bath = st.slider("Half Bathrooms", 0, 2, 0)
        with col_f:
            garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)

        garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 480, step=20)

        # ---- Age & Condition ----
        st.markdown('<div class="input-panel-header">🏗️ Age & Condition</div>', unsafe_allow_html=True)

        current_year = datetime.datetime.now().year
        col_g, col_h = st.columns(2)
        with col_g:
            year_built = st.slider("Year Built", 1872, current_year, 2000)
        with col_h:
            year_remod = st.slider("Year Remodeled", 1950, current_year, 2005)

        col_i, col_j = st.columns(2)
        with col_i:
            roof_matl = st.selectbox("Roof Material", ROOF_MATERIALS, index=0)
        with col_j:
            foundation = st.selectbox("Foundation Type", FOUNDATION_TYPES, index=0)

        # ---- Extras ----
        st.markdown('<div class="input-panel-header">✨ Extras & Quality</div>', unsafe_allow_html=True)

        col_k, col_l = st.columns(2)
        with col_k:
            kitchen_qual = st.selectbox("Kitchen Quality", list(KITCHEN_QUALITY_MAP.keys()), index=1)
        with col_l:
            exterior = st.selectbox("Exterior Covering", EXTERIOR_OPTIONS, index=0)

        col_m, col_n, col_o, col_p = st.columns(4)
        with col_m:
            has_fireplace = st.checkbox("Fireplace", value=True)
        with col_n:
            has_pool = st.checkbox("Pool")
        with col_o:
            has_porch = st.checkbox("Porch", value=True)
        with col_p:
            has_central_air = st.checkbox("Central Air", value=True)

        st.markdown("---")

        # ---- PREDICT BUTTON ----
        predict_clicked = st.button("🔍  Estimate Market Value", use_container_width=True)

    # ------------------------------------------------------------------
    # RIGHT COLUMN — Results Panel
    # ------------------------------------------------------------------
    with right_col:
        if predict_clicked and MODEL_READY:
            with st.spinner("🧠 Calculating your home's value..."):
                # Build the raw feature dictionary matching Ames column names
                input_dict = {
                    "Neighborhood": neighborhood,
                    "HouseStyle": house_style,
                    "OverallQual": overall_qual,
                    "OverallCond": 5,
                    "GrLivArea": gr_liv_area,
                    "TotalBsmtSF": total_bsmt_sf,
                    "BedroomAbvGr": bedrooms,
                    "FullBath": full_bath,
                    "HalfBath": half_bath,
                    "BsmtFullBath": 1 if total_bsmt_sf > 0 else 0,
                    "BsmtHalfBath": 0,
                    "GarageCars": garage_cars,
                    "GarageArea": garage_area,
                    "GarageYrBlt": year_built,
                    "GarageType": "Attchd" if garage_cars > 0 else "None",
                    "GarageFinish": "Unf" if garage_cars > 0 else "None",
                    "GarageQual": "TA" if garage_cars > 0 else "None",
                    "GarageCond": "TA" if garage_cars > 0 else "None",
                    "YearBuilt": year_built,
                    "YearRemodAdd": year_remod,
                    "RoofMatl": roof_matl,
                    "Foundation": foundation,
                    "KitchenQual": KITCHEN_QUALITY_MAP[kitchen_qual],
                    "KitchenAbvGr": 1,
                    "Fireplaces": 1 if has_fireplace else 0,
                    "FireplaceQu": "Gd" if has_fireplace else "None",
                    "PoolArea": 200 if has_pool else 0,
                    "PoolQC": "Gd" if has_pool else "None",
                    "OpenPorchSF": 50 if has_porch else 0,
                    "EnclosedPorch": 0,
                    "3SsnPorch": 0,
                    "ScreenPorch": 0,
                    "CentralAir": "Y" if has_central_air else "N",
                    # Sensible defaults for remaining features
                    "MSSubClass": 60 if "2" in house_style else 20,
                    "MSZoning": "RL",
                    "LotFrontage": 70,
                    "LotArea": 9000,
                    "Street": "Pave",
                    "Alley": "None",
                    "LotShape": "Reg",
                    "LandContour": "Lvl",
                    "Utilities": "AllPub",
                    "LotConfig": "Inside",
                    "LandSlope": "Gtl",
                    "Condition1": "Norm",
                    "Condition2": "Norm",
                    "BldgType": "1Fam",
                    "RoofStyle": "Gable",
                    "Exterior1st": exterior,
                    "Exterior2nd": exterior,
                    "MasVnrType": "None",
                    "MasVnrArea": 0,
                    "ExterQual": "TA",
                    "ExterCond": "TA",
                    "BsmtQual": "TA" if total_bsmt_sf > 0 else "None",
                    "BsmtCond": "TA" if total_bsmt_sf > 0 else "None",
                    "BsmtExposure": "No" if total_bsmt_sf > 0 else "None",
                    "BsmtFinType1": "Unf" if total_bsmt_sf > 0 else "None",
                    "BsmtFinSF1": int(total_bsmt_sf * 0.4),
                    "BsmtFinType2": "Unf",
                    "BsmtFinSF2": 0,
                    "BsmtUnfSF": int(total_bsmt_sf * 0.6),
                    "Heating": "GasA",
                    "HeatingQC": "Ex",
                    "Electrical": "SBrkr",
                    "1stFlrSF": gr_liv_area if "1" in house_style else int(gr_liv_area * 0.55),
                    "2ndFlrSF": 0 if "1" in house_style else int(gr_liv_area * 0.45),
                    "LowQualFinSF": 0,
                    "TotRmsAbvGrd": bedrooms + 2,
                    "Functional": "Typ",
                    "WoodDeckSF": 100,
                    "MiscFeature": "None",
                    "MiscVal": 0,
                    "MoSold": 6,
                    "YrSold": 2010,
                    "SaleType": "WD",
                    "SaleCondition": "Normal",
                    "Fence": "None",
                    "PavedDrive": "Y",
                }

                # Run prediction
                from utils.predict import predict_price
                model, feature_names = load_model_cached()
                predicted_price, aligned_df = predict_price(input_dict, model, feature_names)

                # Confidence range (±8%)
                price_low = predicted_price * 0.92
                price_high = predicted_price * 1.08

            # ==============================================================
            # Display Results
            # ==============================================================

            # --- Big Price Card ---
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; padding: 32px;">
                <p class="price-label">Estimated Market Value</p>
                <p class="price-display">${predicted_price:,.0f}</p>
                <p class="price-range">Confidence range: ${price_low:,.0f} — ${price_high:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Three Mini-Metric Cards ---
            price_per_sqft = predicted_price / max(gr_liv_area, 1)

            # Luxury tier
            luxury_score = overall_qual * 2
            if luxury_score >= 16:
                tier = "🏆 Premium"
            elif luxury_score >= 12:
                tier = "⭐ Upper-Mid"
            elif luxury_score >= 8:
                tier = "🏠 Mid-Range"
            else:
                tier = "🔑 Economy"

            # Neighborhood comparison
            train_data = load_training_data()
            if train_data is not None and neighborhood in train_data["Neighborhood"].values:
                hood_median = train_data[train_data["Neighborhood"] == neighborhood]["SalePrice"].median()
                hood_diff = ((predicted_price - hood_median) / hood_median) * 100
                hood_text = f"{hood_diff:+.1f}%" if abs(hood_diff) < 200 else "—"
            else:
                hood_text = "—"
                hood_median = predicted_price

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-metric-value">${price_per_sqft:,.0f}</div>
                    <div class="mini-metric-label">Price / sqft</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-metric-value">{hood_text}</div>
                    <div class="mini-metric-label">vs. Neighborhood Avg</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-metric-value">{tier}</div>
                    <div class="mini-metric-label">Luxury Tier</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- SHAP Explainability Chart ---
            try:
                from utils.shap_explainer import get_shap_explanation, plot_shap_bar

                log_price = np.log1p(predicted_price)
                shap_df = get_shap_explanation(
                    model, aligned_df, feature_names,
                    top_n=10, predicted_log_price=log_price,
                )
                fig_shap = plot_shap_bar(shap_df)
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")

            # --- Market Context ---
            if train_data is not None and neighborhood in train_data["Neighborhood"].values:
                hood_data = train_data[train_data["Neighborhood"] == neighborhood]["SalePrice"]
                p25, p75 = int(hood_data.quantile(0.25)), int(hood_data.quantile(0.75))

                st.markdown(f"""
                <div class="metric-card">
                    <p class="section-header">📍 Market Context — {neighborhood}</p>
                    <p style="color:#374151; font-size:0.95rem;">
                        Similar homes in <strong>{neighborhood}</strong> typically sell between
                        <strong>${p25:,}</strong> and <strong>${p75:,}</strong>.<br>
                        Median price: <strong>${int(hood_median):,}</strong>
                        &nbsp;·&nbsp; Homes analyzed: <strong>{len(hood_data)}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # --- Download Report ---
            report_text = (
                f"REAL ESTATE VALUE ESTIMATOR — VALUATION REPORT\n"
                f"{'=' * 50}\n"
                f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}\n\n"
                f"PROPERTY DETAILS\n"
                f"  Neighborhood      : {neighborhood}\n"
                f"  House Style       : {house_style}\n"
                f"  Overall Quality   : {overall_qual}/10\n"
                f"  Living Area       : {gr_liv_area} sqft\n"
                f"  Basement Area     : {total_bsmt_sf} sqft\n"
                f"  Bedrooms          : {bedrooms}\n"
                f"  Bathrooms (Full)  : {full_bath}\n"
                f"  Garage (cars)     : {garage_cars}\n"
                f"  Year Built        : {year_built}\n"
                f"  Year Remodeled    : {year_remod}\n\n"
                f"VALUATION\n"
                f"  Estimated Value   : ${predicted_price:,.0f}\n"
                f"  Confidence Range  : ${price_low:,.0f} - ${price_high:,.0f}\n"
                f"  Price per sqft    : ${price_per_sqft:,.0f}\n"
                f"  Luxury Tier       : {tier}\n\n"
                f"{'=' * 50}\n"
                f"Generated by Real Estate Value Estimator (XGBoost + SHAP)\n"
            )

            st.download_button(
                "📄  Download Valuation Report",
                data=report_text,
                file_name="valuation_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        elif predict_clicked and not MODEL_READY:
            st.error("Model not found. Please train the model first (see instructions above).")

        elif not predict_clicked:
            # Placeholder when no prediction has been made yet
            st.markdown("""
            <div class="metric-card" style="text-align: center; padding: 60px 30px;">
                <p style="font-size: 3rem; margin-bottom: 12px;">🏠</p>
                <p style="font-size: 1.2rem; font-weight: 600; color: #1a1a2e; margin-bottom: 8px;">
                    Your Valuation Will Appear Here
                </p>
                <p style="color: #6b7280; font-size: 0.95rem; max-width: 380px; margin: 0 auto;">
                    Fill in the property details on the left, then click
                    <strong>Estimate Market Value</strong> to get an AI-powered valuation
                    with full explainability.
                </p>
            </div>
            """, unsafe_allow_html=True)


