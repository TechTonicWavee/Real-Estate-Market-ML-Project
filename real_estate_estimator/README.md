# 🏠 Real Estate Value Estimator

An AI-powered property valuation system built with **XGBoost**, **SHAP explainability**, and a polished **Streamlit** dashboard. This is not a basic house price predictor — it's a professional-grade intelligent valuation tool that handles real-world messy data, performs deep feature engineering, and explains _why_ a price was calculated.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 ML Prediction | XGBoost regressor trained on 80+ features from the Ames Housing Dataset |
| 🔍 Explainability | SHAP-based breakdown showing which factors raised/lowered the price |
| 📊 Market Trends | Interactive Plotly charts for neighborhood prices, size vs price, trends over time |
| 💎 Premium UI | Custom-styled Streamlit dashboard with gradient buttons, card components, dark sidebar |
| 📄 Report Export | Downloadable valuation report for any property estimate |
| ⚡ Fast Inference | Cached model loading — instant predictions after first load |

---

## 📁 Project Structure

```
real_estate_estimator/
│
├── app.py                  ← Main Streamlit app (entry point)
├── requirements.txt        ← All Python dependencies
├── README.md               ← You are here
│
├── model/
│   ├── train_model.py      ← Full training pipeline script
│   ├── xgb_model.pkl       ← Saved trained XGBoost model (generated)
│   └── feature_names.pkl   ← Saved feature column list (generated)
│
├── data/
│   └── train.csv           ← Ames Housing dataset (YOU download this)
│
├── utils/
│   ├── preprocessing.py    ← Data cleaning + feature engineering
│   ├── predict.py          ← Prediction logic using loaded model
│   └── shap_explainer.py   ← SHAP value calculation + Plotly charts
│
└── assets/
    └── style.css           ← Custom CSS for the Streamlit UI
```

---

## 🚀 Quick Start

### 1. Download the Dataset

Go to the Kaggle competition page and download `train.csv`:

👉 **https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data**

Place the file at:
```
real_estate_estimator/data/train.csv
```

### 2. Install Dependencies

```bash
cd real_estate_estimator
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python model/train_model.py
```

This will:
- Load and clean the Ames Housing data
- Engineer 15+ composite features
- Train an XGBoost regressor with early stopping
- Print evaluation metrics (RMSE, MAE, R²)
- Save the model to `model/xgb_model.pkl`

### 4. Launch the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🧪 How It Works

### Feature Engineering (Domain Intelligence)

The system creates composite features encoding real-estate domain knowledge:

| Engineered Feature | Formula |
|---|---|
| `HouseAge` | Current year − Year built |
| `TotalSF` | Basement finished SF + 1st floor + 2nd floor |
| `TotalBathrooms` | Full baths + 0.5 × half baths (incl. basement) |
| `OverallLuxury` | Weighted composite of quality ratings |
| `HasPool`, `HasGarage`, etc. | Boolean flags derived from area columns |

### Model

- **XGBoost** with 500 estimators, learning rate 0.05, early stopping at 50 rounds
- Target is log-transformed (`log1p`) to handle right-skewed prices
- Predictions are reverse-transformed (`expm1`) back to real dollars

### Explainability

- **SHAP TreeExplainer** computes per-feature contributions
- Log-scale SHAP values are converted to approximate dollar impacts
- Green bars = features that **increased** the estimated value
- Red bars = features that **decreased** it

---

## 📊 Market Trends Tab

The second tab provides four interactive Plotly charts:

1. **Price by Neighborhood** — sorted horizontal bar chart with price-tier coloring
2. **Price vs Size** — scatter plot colored by Overall Quality
3. **Price Over Time** — year-built trend with rolling average
4. **Feature Importance** — top 15 XGBoost feature importances

---

## 🛠 Tech Stack

- **Streamlit** — UI framework
- **XGBoost** — gradient-boosted tree model
- **SHAP** — model explainability
- **Plotly** — all interactive charts
- **Pandas / NumPy / Scikit-learn** — data processing
- **Joblib** — model persistence

---

## 📝 License

This project is for educational and portfolio purposes. The Ames Housing Dataset is provided under the Kaggle competition terms.
