import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import joblib
from feature_engineer import engineer_features_for_date, get_feature_names

# Page configuration
st.set_page_config(
    page_title="Lahore PM2.5 Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

@st.cache_resource
def load_data():
    """Load historical data for lag/rolling features."""
    df = pd.read_csv("../data/processed/merged_data_extended.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_scaler():
    """Load feature scaler."""
    return joblib.load("../models/scaler.pkl")

@st.cache_resource
def load_model(model_name):
    """Load trained model."""
    return joblib.load(f"../models/{model_name}_model.pkl")

@st.cache_data
def load_model_performance():
    """Load model performance metrics from training results."""
    try:
        with open("../models/training_results.json", "r") as f:
            results = json.load(f)
        return results
    except:
        return None

def get_aqi_category(pm25_value):
    """Get AQI category and color based on PM2.5 value."""
    if pm25_value <= 12.0:
        return {
            'category': 'Good',
            'color': '#00E400',
            'health_implications': 'Air quality is satisfactory, and air pollution poses little or no risk.'
        }
    elif pm25_value <= 35.4:
        return {
            'category': 'Moderate',
            'color': '#FFFF00',
            'health_implications': 'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.'
        }
    elif pm25_value <= 55.4:
        return {
            'category': 'Unhealthy for Sensitive Groups',
            'color': '#FF7E00',
            'health_implications': 'Members of sensitive groups may experience health effects. The general public is less likely to be affected.'
        }
    elif pm25_value <= 150.4:
        return {
            'category': 'Unhealthy',
            'color': '#FF0000',
            'health_implications': 'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.'
        }
    elif pm25_value <= 250.4:
        return {
            'category': 'Very Unhealthy',
            'color': '#8F3F97',
            'health_implications': 'Health alert: The risk of health effects is increased for everyone.'
        }
    else:
        return {
            'category': 'Hazardous',
            'color': '#7E0023',
            'health_implications': 'Health warning of emergency conditions: everyone is more likely to be affected.'
        }

def main():
    # Header
    st.markdown('<p class="main-header">🌫️ Lahore PM2.5 Forecasting Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Predict Tomorrow's Air Quality Using Machine Learning")
    
    # Load data and models
    try:
        df = load_data()
        scaler = load_scaler()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Get date range
    min_date = df['date'].iloc[30].date()
    max_date = df['date'].max().date()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "Lasso Regression": "lasso",
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "Random Forest": "random_forest",
            "Gradient Boosting": "gradient_boosting",
            "Ridge Regression": "ridge",
            "SVR": "svr"
        }
        
        selected_model_name = st.selectbox(
            "🤖 Select ML Model",
            options=list(model_options.keys()),
            index=0,
            help="Choose the machine learning model for prediction"
        )
        
        selected_model = model_options[selected_model_name]
        
        st.markdown("---")
        
        # Reference date for historical data
        st.subheader("📅 Reference Date")
        st.caption("Used to fetch historical data for lag/rolling features")
        
        selected_date = st.date_input(
            "Select reference date",
            value=datetime(2024, 11, 15).date(),
            min_value=min_date,
            max_value=max_date,
            help="This date provides historical context (lag features, rolling averages)"
        )
        
    
    # Main input form
    st.markdown("### 📝 Enter Today's Pollution & Weather Data")
    st.caption("These are the most important features for prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌫️ **Air Quality**")
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=150.0, step=10.0,
                               help="Particulate Matter 10 micrometers or less")
        ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, max_value=300.0, value=100.0, step=10.0,
                                help="Ground-level ozone concentration")
        carbon_monoxide = st.number_input("Carbon Monoxide (µg/m³)", min_value=0.0, max_value=10000.0, value=3000.0, step=100.0,
                                         help="CO concentration")
        sulphur_dioxide = st.number_input("Sulphur Dioxide (µg/m³)", min_value=0.0, max_value=150.0, value=30.0, step=5.0,
                                         help="SO2 concentration")
        nitrogen_dioxide = st.number_input("Nitrogen Dioxide (µg/m³)", min_value=0.0, max_value=300.0, value=80.0, step=10.0,
                                          help="NO2 concentration")
    
    with col2:
        st.markdown("#### 🌤️ **Weather**")
        tmax = st.number_input("Max Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=1.0)
        relative_humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=5.0)
        prcp = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        wspd = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        pres = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0, step=1.0)
        
        st.markdown("#### 🔥 **Fire Data**")
        fire_frp_max = st.number_input("Fire FRP Max", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                                       help="Maximum Fire Radiative Power")
        fire_frp_total = st.number_input("Fire FRP Total", min_value=0.0, max_value=500.0, value=0.0, step=10.0,
                                        help="Total Fire Radiative Power")
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 Predict Tomorrow's PM2.5", type="primary", use_container_width=True)
    
    # Sidebar - Model info
    with st.sidebar:
        st.markdown("---")
        with st.expander("📊 Model Performance", expanded=False):
            model_perf = load_model_performance()
            if model_perf:
                # Find the selected model's metrics
                all_results = (model_perf.get('classical_results', []) + 
                             model_perf.get('ensemble_results', []))
                
                for result in all_results:
                    if result['model'].lower().replace(' ', '_') == selected_model:
                        st.metric("R² Score", f"{result['r2']:.4f}")
                        st.metric("MAE", f"{result['mae']:.2f} µg/m³")
                        st.metric("RMSE", f"{result['rmse']:.2f} µg/m³")
                        st.metric("MAPE", f"{result['mape']:.2f}%")
                        break
    
    # Make prediction
    if predict_button:
        try:
            with st.spinner("🔄 Engineering features and making prediction..."):
                # Create a temporary dataframe with user inputs
                user_data = df.copy()
                
                # Update the selected date row with user inputs
                date_idx = user_data[user_data['date'] == pd.to_datetime(selected_date)].index
                
                if len(date_idx) == 0:
                    st.error(f"Date {selected_date} not found in dataset!")
                    st.stop()
                
                date_idx = date_idx[0]
                
                # Update with user inputs
                user_data.loc[date_idx, 'pm10'] = pm10
                user_data.loc[date_idx, 'ozone'] = ozone
                user_data.loc[date_idx, 'carbon_monoxide'] = carbon_monoxide
                user_data.loc[date_idx, 'sulphur_dioxide'] = sulphur_dioxide
                user_data.loc[date_idx, 'nitrogen_dioxide'] = nitrogen_dioxide
                user_data.loc[date_idx, 'tmax'] = tmax
                user_data.loc[date_idx, 'relative_humidity_2m'] = relative_humidity
                user_data.loc[date_idx, 'prcp'] = prcp
                user_data.loc[date_idx, 'wspd'] = wspd
                user_data.loc[date_idx, 'pres'] = pres
                user_data.loc[date_idx, 'fire_frp_max'] = fire_frp_max
                user_data.loc[date_idx, 'fire_frp_total'] = fire_frp_total
                
                # Calculate derived weather features
                tmin = user_data.loc[date_idx, 'tmin']  # Keep original tmin
                tavg = (tmax + tmin) / 2
                user_data.loc[date_idx, 'tavg'] = tavg
                
                # Engineer features
                features = engineer_features_for_date(user_data, selected_date)
                
                # Get feature names in correct order
                feature_names = get_feature_names()
                feature_vector = features[feature_names].values.reshape(1, -1)
                
                # Handle NaN values
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                
                # Scale features
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Load model and predict
                model = load_model(selected_model)
                predicted_pm25 = model.predict(feature_vector_scaled)[0]
                
                # Get actual value for next day if available
                next_day = pd.to_datetime(selected_date) + timedelta(days=1)
                actual_pm25 = None
                if next_day in df['date'].values:
                    actual_pm25 = df[df['date'] == next_day]['pm2_5_mean'].values[0]
                
                # Store result
                st.session_state.prediction_result = {
                    'prediction_date': selected_date,
                    'target_date': next_day.date(),
                    'predicted_pm25': float(predicted_pm25),
                    'actual_pm25': float(actual_pm25) if actual_pm25 is not None else None,
                    'model_name': selected_model_name,
                    'user_inputs': {
                        'pm10': pm10, 'ozone': ozone, 'co': carbon_monoxide,
                        'so2': sulphur_dioxide, 'no2': nitrogen_dioxide,
                        'tmax': tmax, 'humidity': relative_humidity
                    }
                }
                
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Display results
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Main prediction display
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("#### 📅 Today's Date")
            st.markdown(f"### {result['prediction_date'].strftime('%b %d, %Y')}")
            st.caption("(Reference date)")
        
        with col2:
            st.markdown("#### 🎯 Tomorrow's Date")
            st.markdown(f"### {result['target_date'].strftime('%b %d, %Y')}")
            st.caption("(Predicted date)")
        
        with col3:
            st.markdown("#### 🤖 Model")
            st.markdown(f"### {result['model_name']}")
        
        st.markdown("---")
        
        # Prediction result with AQI category
        predicted_pm25 = result['predicted_pm25']
        aqi_info = get_aqi_category(predicted_pm25)
        
        st.markdown("### 🌡️ Predicted PM2.5 Concentration")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="background-color: {aqi_info['color']}; padding: 2rem; border-radius: 10px; text-align: center;">
                <h1 style="color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                    {predicted_pm25:.1f} µg/m³
                </h1>
                <h3 style="color: white; margin-top: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                    {aqi_info['category']}
                </h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <strong>Health Implications:</strong><br>
                {aqi_info['health_implications']}
            </div>
            """, unsafe_allow_html=True)
        
        # Actual vs Predicted comparison (if actual available)
        if result['actual_pm25'] is not None:
            st.markdown("---")
            st.markdown("### 📊 Prediction vs Actual")
            
            actual_pm25 = result['actual_pm25']
            error = predicted_pm25 - actual_pm25
            error_pct = (error / actual_pm25) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted", f"{predicted_pm25:.1f} µg/m³")
            with col2:
                st.metric("Actual", f"{actual_pm25:.1f} µg/m³")
            with col3:
                st.metric("Error", f"{error:.1f} µg/m³", delta=f"{error_pct:.1f}%")
            with col4:
                accuracy = 100 - abs(error_pct)
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            # Comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Predicted',
                x=['PM2.5'],
                y=[predicted_pm25],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Actual',
                x=['PM2.5'],
                y=[actual_pm25],
                marker_color='orange'
            ))
            
            fig.update_layout(
                title="Predicted vs Actual PM2.5",
                yaxis_title="PM2.5 (µg/m³)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show user inputs
        st.markdown("---")
        with st.expander("📊 Your Input Values", expanded=False):
            inputs = result['user_inputs']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PM10", f"{inputs['pm10']:.1f} µg/m³")
                st.metric("Ozone", f"{inputs['ozone']:.1f} µg/m³")
                st.metric("CO", f"{inputs['co']:.1f} µg/m³")
            with col2:
                st.metric("SO2", f"{inputs['so2']:.1f} µg/m³")
                st.metric("NO2", f"{inputs['no2']:.1f} µg/m³")
            with col3:
                st.metric("Max Temp", f"{inputs['tmax']:.1f}°C")
                st.metric("Humidity", f"{inputs['humidity']:.1f}%")
        
        # Download results
        st.markdown("---")
        result_df = pd.DataFrame([{
            'Today': result['prediction_date'],
            'Tomorrow': result['target_date'],
            'Predicted PM2.5': result['predicted_pm25'],
            'Actual PM2.5': result['actual_pm25'],
            'Model': result['model_name'],
            'AQI Category': aqi_info['category'],
            **result['user_inputs']
        }])
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"pm25_prediction_{result['target_date']}.csv",
            mime="text/csv"
        )
    
    else:
        # Initial state - show instructions
        st.markdown("""
        <div class="info-box">
            <h4>👋 Welcome to Lahore PM2.5 Forecasting Dashboard!</h4>
            <p>Predict tomorrow's air quality by entering today's pollution and weather conditions.</p>
            <br>
            <h5>📋 How to use:</h5>
            <ol>
                <li><strong>Select a model</strong> from the sidebar (Lasso, XGBoost, etc.)</li>
                <li><strong>Choose a reference date</strong> (provides historical context for predictions)</li>
                <li><strong>Enter today's values</strong> for key pollution and weather parameters</li>
                <li><strong>Click "Predict"</strong> to forecast tomorrow's PM2.5 level</li>
            </ol>
            <br>
            <h5>🔬 How it works:</h5>
            <ul>
                <li>You provide <strong>current conditions</strong> (PM10, Ozone, Temperature, etc.)</li>
                <li>The app automatically fetches <strong>historical data</strong> for lag features (yesterday's values, 7-day averages, etc.)</li>
                <li>Machine learning model <strong>predicts tomorrow's PM2.5</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample values
        st.markdown("---")
        st.markdown("### 💡 Sample Values for Testing")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **Winter Smog (High Pollution):**
            - PM10: 250-350 µg/m³
            - Ozone: 80-120 µg/m³
            - Temperature: 10-15°C
            - Wind Speed: 0.5-1.5 m/s
            """)
        with col2:
            st.success("""
            **Summer Clear Day (Low Pollution):**
            - PM10: 40-80 µg/m³
            - Ozone: 150-200 µg/m³
            - Temperature: 30-38°C
            - Wind Speed: 4-8 m/s
            """)

if __name__ == "__main__":
    main()
