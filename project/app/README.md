# Lahore PM2.5 Forecasting - Streamlit App

## Installation

1. Install required packages:
```bash
pip install streamlit plotly pandas numpy scikit-learn xgboost lightgbm
```

## Running the App

1. Navigate to the app directory:
```bash
cd d:\SEM5\ML\PROJECT\lahore-aqi-forecast\project\app
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The app will open in your default browser at `http://localhost:8501`

## Features

- **Multiple ML Models**: Choose from 7 trained models (Lasso, XGBoost, LightGBM, Random Forest, etc.)
- **Date Selection**: Predict for any date from Feb 2023 to Dec 2025
- **Automatic Feature Engineering**: Generates 93 features automatically from selected date
- **AQI Categories**: Color-coded health implications
- **Model Performance**: View accuracy metrics for each model
- **Actual vs Predicted**: Compare predictions with actual values (for historical dates)
- **Downloadable Results**: Export predictions as CSV

## Data Sources

- **Historical Data**: Jan 1, 2023 - May 1, 2025 (real data)
- **Projected Data**: May 2, 2025 - Dec 31, 2025 (averaged from 2023-2024)

## Files

- `app.py` - Main Streamlit dashboard
- `predictor.py` - Prediction handler and model loader
- `feature_engineer.py` - Feature engineering pipeline
