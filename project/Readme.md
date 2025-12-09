# Lahore AQI Forecast Project

This project predicts the next day's PM2.5 levels for Lahore using machine learning models trained on historical air quality, weather, and fire data (Jan 2023 - May 2025).

## Quick Start (Using Pre-trained Models)

For groupmates who just want to run the prediction app:

```bash
# 1. Clone the repository
git clone https://github.com/Abdullah-Farooq-5/lahore-aqi-forecast.git
cd lahore-aqi-forecast/project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
cd app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Enter today's pollution and weather values to get tomorrow's PM2.5 prediction!

---

## Full Pipeline (Training from Scratch)

If you want to retrain models or understand the complete workflow:

### Step 1: Data Preprocessing

```bash
# Merge raw data files
python src/merge.py

# Clean and preprocess the merged data
python src/data_preprocessing.py

# Engineer features (lag, rolling means, interactions, etc.)
python src/feature_engineering.py
```

**Output Files:**
- `data/processed/merged_data.csv` - Combined raw data
- `data/processed/merged_data_clean.csv` - Cleaned data
- `data/processed/features_engineered.csv` - 93 engineered features

### Step 2: Extend Dataset (Optional)

```bash
# Extend data to Dec 2025 by averaging 2023-2024 data
python extend_dataset.py
```

**Output:** `data/processed/merged_data_extended.csv` (1,097 days)

### Step 3: Train Models

```bash
python src/train_models.py
```

**Output:**
- 7 trained models in `models/` (Random Forest, Lasso, Ridge, XGBoost, LightGBM, SVR, Gradient Boosting)
- `models/scaler.pkl` - Feature scaler
- `models/training_results.json` - Performance metrics
- `models/model_comparison.csv` - Model comparison

### Step 4: Run the App

```bash
cd app
streamlit run app.py
```

---

## Project Structure

```
project/
├── data/
│   ├── raw/                     # Original datasets
│   │   ├── aqi.csv             # Air quality data
│   │   ├── weather.csv         # Weather data
│   │   └── fire.csv            # Fire incidents data
│   └── processed/               # Processed datasets
│       ├── merged_data.csv
│       ├── merged_data_clean.csv
│       ├── features_engineered.csv
│       └── merged_data_extended.csv
├── src/
│   ├── merge.py                # Merge raw data files
│   ├── data_preprocessing.py   # Clean and preprocess data
│   ├── feature_engineering.py  # Create 93 engineered features
│   └── train_models.py         # Train 7 ML models
├── app/
│   ├── app.py                  # Streamlit dashboard
│   └── feature_engineer.py     # Feature engineering module
├── models/                      # Trained models (.pkl files)
├── extend_dataset.py           # Extend data to Dec 2025
├── featureimp.py              # Analyze feature importance
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Model Performance

| Model              | Train R² | Val R²  | Test R²  | Test MAE | Test RMSE |
|--------------------|----------|---------|----------|----------|-----------|
| **Lasso**          | 0.8368   | 0.8294  | 0.8372   | 16.58    | 21.85     |
| Ridge              | 0.8379   | 0.8307  | 0.8371   | 16.67    | 21.86     |
| Random Forest      | 0.9874   | 0.8149  | 0.8216   | 17.42    | 22.87     |
| Gradient Boosting  | 0.9302   | 0.8035  | 0.8143   | 17.30    | 23.33     |
| XGBoost            | 0.9952   | 0.7934  | 0.8063   | 17.48    | 23.82     |
| LightGBM           | 0.9946   | 0.7897  | 0.8085   | 17.62    | 23.69     |
| SVR                | 0.8131   | 0.7929  | 0.8039   | 18.19    | 23.97     |

**Best Model:** Lasso (R² = 0.837, MAE = 16.58)

---

## Features (93 Total)

### Feature Categories:
- **Temporal Features (18):** Hour, day, month, season, day of week, week of year, etc.
- **Lag Features (24):** 1, 2, 3, 7, 14, 30-day lags for pollutants
- **Rolling Mean Features (28):** 3, 7, 14, 30-day rolling averages
- **Weather-derived (6):** Temp-humidity interaction, pressure change, wind chill, heat index
- **Interaction Features (5):** PM2.5×PM10, PM10×Ozone, Temperature×Humidity, etc.
- **Ratio Features (2):** PM2.5/PM10 ratio, AQI ratio
- **Fire Features (3):** Confidence, brightness, frp (fire radiative power)
- **Extreme Pollution Indicators (6):** Binary flags for severe pollution events

---

## How the App Works

1. **User Input:** Enter today's pollution values (PM10, Ozone, CO, SO2, NO2) and weather data (Temperature, Humidity, Wind Speed, Pressure, Fire data)

2. **Reference Date:** Select a date from historical data to fetch lag and rolling mean features

3. **Feature Engineering:** The app automatically:
   - Updates the selected date with your input values
   - Engineers 93 features (temporal, lag, rolling, interactions, etc.)
   - Scales features using the trained scaler

4. **Prediction:** Selected model predicts tomorrow's PM2.5 level

5. **Output:** 
   - Tomorrow's PM2.5 prediction
   - AQI category (Good, Moderate, Unhealthy, etc.)
   - Health implications
   - Comparison chart (actual vs predicted for reference date)

---

## Troubleshooting

### `streamlit: command not found`
```bash
pip install streamlit plotly
```

### `ModuleNotFoundError: No module named 'xgboost'`
```bash
pip install -r requirements.txt
```

### Models not loading
Ensure you're running the app from the `app/` directory:
```bash
cd app
streamlit run app.py
```

### Data file not found
Make sure `merged_data_extended.csv` exists in `data/processed/`. If not, run:
```bash
python extend_dataset.py
```

---

## Dependencies

- Python >= 3.8
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- streamlit >= 1.28.0
- plotly >= 5.17.0
- joblib >= 1.3.0

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Contributors

- Abdullah Farooq
- [Your Groupmate's Name]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
