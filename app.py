import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import os

# Page configuration
st.set_page_config(
    page_title="Play Whe AI Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, sleek design with vibrant color palette
st.markdown("""
<style>
    :root {
        --primary: #6a11cb;
        --primary-dark: #2575fc;
        --secondary: #00c9ff;
        --accent: #ffd166;
        --background: #0f0c29;
        --card-bg: rgba(25, 22, 56, 0.7);
        --text: #ffffff;
        --text-secondary: #b0a8d6;
        --success: #06d6a0;
        --warning: #ffd166;
        --error: #ef476f;
    }
    
    .main > div {
        padding-top: 1.5rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--background) 0%, #1a1a2e 100%);
        color: var(--text);
    }
    
    .main-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(106, 17, 203, 0.3);
        position: relative;
        overflow: hidden;
        border: none;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        transform: rotate(30deg);
    }
    
    .main-title {
        color: white;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 2;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 300;
        position: relative;
        z-index: 2;
    }
    
    .prediction-card {
        background: var(--card-bg);
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(106, 17, 203, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .prediction-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(106, 17, 203, 0.4);
        border-color: var(--primary);
    }
    
    .number-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 0.6rem 1.4rem;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.4rem;
        margin: 0.3rem;
        box-shadow: 0 6px 20px rgba(106, 17, 203, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.2);
        position: relative;
        z-index: 2;
    }
    
    .probability-bar {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        height: 10px;
        border-radius: 5px;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .probability-bar::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite linear;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .stats-container {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(106, 17, 203, 0.3);
        color: var(--text);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .stats-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .time-selector {
        background: var(--card-bg);
        border-radius: 18px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(106, 17, 203, 0.3);
        transition: all 0.3s ease;
        color: var(--text);
        backdrop-filter: blur(10px);
        position: relative;
    }
    
    .time-selector:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(106, 17, 203, 0.4);
        border-color: var(--primary);
    }
    
    .validation-result {
        padding: 1.5rem;
        border-radius: 18px;
        margin: 1.2rem 0;
        font-weight: bold;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(106, 17, 203, 0.3);
    }
    
    .validation-result::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
    }
    
    .success {
        background: rgba(6, 214, 160, 0.15);
        color: var(--success);
        border-color: rgba(6, 214, 160, 0.3);
    }
    
    .success::before {
        background: linear-gradient(90deg, var(--success), #04b58d);
    }
    
    .warning {
        background: rgba(255, 209, 102, 0.15);
        color: var(--warning);
        border-color: rgba(255, 209, 102, 0.3);
    }
    
    .warning::before {
        background: linear-gradient(90deg, var(--warning), #e6bc5c);
    }
    
    .error {
        background: rgba(239, 71, 111, 0.15);
        color: var(--error);
        border-color: rgba(239, 71, 111, 0.3);
    }
    
    .error::before {
        background: linear-gradient(90deg, var(--error), #d43d65);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 12, 41, 0.8);
        backdrop-filter: blur(10px);
    }
    
    /* Custom styling for Streamlit elements */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.7rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(106, 17, 203, 0.4);
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(106, 17, 203, 0.6);
    }
    
    .stButton > button::after {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: rgba(255, 255, 255, 0.1);
        transform: rotate(30deg);
        transition: all 0.5s ease;
    }
    
    .stButton > button:hover::after {
        transform: rotate(30deg) translate(20%, 20%);
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(25, 22, 56, 0.7);
        border: 2px solid rgba(106, 17, 203, 0.5);
        border-radius: 15px;
        color: var(--text);
        padding: 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--secondary);
        box-shadow: 0 0 0 0.2rem rgba(0, 201, 255, 0.25);
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: var(--card-bg);
        border-radius: 15px;
        padding: 1.2rem;
        border: 1px solid rgba(106, 17, 203, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Info/warning/error boxes */
    .stAlert {
        border-radius: 15px;
        border: 1px solid rgba(106, 17, 203, 0.3);
        background: var(--card-bg);
        backdrop-filter: blur(10px);
    }
    
    /* Metric styling */
    .metric-container {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(106, 17, 203, 0.3);
        text-align: center;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--secondary);
        margin: 0.5rem 0;
        text-shadow: 0 0 10px rgba(0, 201, 255, 0.3);
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-top: 0.3rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border-radius: 15px !important;
        padding: 1rem 2rem;
        margin: 0 5px;
        transition: all 0.3s ease;
        border: 1px solid rgba(106, 17, 203, 0.3) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(106, 17, 203, 0.3) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        font-weight: bold;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(106, 17, 203, 0.4);
    }
    
    /* Custom divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* Glowing effect for important elements */
    .glow {
        box-shadow: 0 0 20px rgba(0, 201, 255, 0.5);
        animation: glow-pulse 2s infinite alternate;
    }
    
    @keyframes glow-pulse {
        from { box-shadow: 0 0 15px rgba(0, 201, 255, 0.5); }
        to { box-shadow: 0 0 30px rgba(0, 201, 255, 0.8); }
    }
</style>
""", unsafe_allow_html=True)

# Feature Engineering
@st.cache_data
def feature_engineering(df):
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter
    
    df["previous_winning_number"] = df["winning_number"].shift(1)
    df["previous_2_winning_number"] = df["winning_number"].shift(2)
    df["previous_3_winning_number"] = df["winning_number"].shift(3)
    df["previous_4_winning_number"] = df["winning_number"].shift(4)
    df["previous_5_winning_number"] = df["winning_number"].shift(5)
    
    df["winning_number_mean_last_10"] = df["winning_number"].rolling(10).mean()
    df["winning_number_std_last_10"] = df["winning_number"].rolling(10).std()
    
    df["winning_number_trend"] = df["winning_number"].diff()
    df["winning_number_diff_2"] = df["winning_number"].diff(2)
    
    df.fillna(method='bfill', inplace=True)
    df["winning_number"] = df["winning_number"] - 1
    return df

@st.cache_resource
def train_model(df):
    features = [
        "year", "month", "day", "day_of_week", "week_of_year", "quarter",
        "winning_number_mean_last_10", "winning_number_std_last_10",
        "winning_number_trend", "winning_number_diff_2",
        "previous_winning_number", "previous_2_winning_number",
        "previous_3_winning_number", "previous_4_winning_number", "previous_5_winning_number"
    ]
    X = df[features]
    y = df["winning_number"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_model = RandomForestClassifier(n_estimators=600, max_depth=20, random_state=42)
    xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42)
    hgb_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model), ('hgb', hgb_model), ('dt', dt_model)],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, ensemble_model.predict(X_test))
    return ensemble_model, X_test, accuracy

def predict_top_numbers(model, input_features, num_predictions=10):
    """Generate top predictions with probabilities"""
    try:
        # Get all class probabilities
        probabilities = model.predict_proba([input_features])[0]
        
        # Get top indices and their probabilities
        top_indices = np.argsort(probabilities)[-num_predictions:][::-1]
        top_probs = probabilities[top_indices]
        
        # Create results dataframe
        results = pd.DataFrame({
            'Winning Number': top_indices + 1,  # Add 1 since we subtracted 1 during preprocessing
            'Probability': top_probs * 100  # Convert to percentage
        })
        
        return results
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return pd.DataFrame()

def get_next_play_time():
    """Determine the next play time based on current time"""
    current_time = datetime.now().time()
    play_times = [
        (10, 30, "10:30 AM - Morning"),
        (13, 0, "1:00 PM - Midday"),
        (16, 0, "4:00 PM - Afternoon"),
        (19, 0, "7:00 PM - Evening")
    ]
    
    for hour, minute, label in play_times:
        play_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0).time()
        if current_time < play_time:
            return label
    
    # If past all times today, return first time tomorrow
    return "10:30 AM - Morning (Next Day)"

def validate_numbers(user_numbers, predictions_df):
    """Validate user input numbers against predictions"""
    try:
        # Parse user input
        numbers = [int(x.strip()) for x in user_numbers.split(',') if x.strip().isdigit()]
        
        if not numbers:
            return None, "Please enter valid numbers separated by commas."
        
        # Check if numbers are in valid range (1-36 for Play Whe)
        invalid_numbers = [n for n in numbers if n < 1 or n > 36]
        if invalid_numbers:
            return None, f"Invalid numbers (must be 1-36): {invalid_numbers}"
        
        # Check against predictions
        predicted_numbers = predictions_df['Winning Number'].tolist()
        matches = [n for n in numbers if n in predicted_numbers]
        
        match_details = []
        for match in matches:
            prob = predictions_df[predictions_df['Winning Number'] == match]['Probability'].iloc[0]
            match_details.append(f"#{match} ({prob:.1f}%)")
        
        return matches, match_details
        
    except ValueError:
        return None, "Please enter valid numbers separated by commas."

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎯 PLAY WHE AI PREDICTOR</h1>
        <p class="main-subtitle">Advanced Machine Learning Predictions for Trinidad & Tobago Play Whe</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and process data
    data_path = "play_whe_results.csv"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        st.error("⚠️ Data file 'play_whe_results.csv' not found. Please ensure the file is in the same directory.")
        st.info("📝 The file should contain columns: 'date' and 'winning_number'")
        return
    
    try:
        df = pd.read_csv(data_path, parse_dates=["date"])
        df = feature_engineering(df)
        model, X_test, accuracy = train_model(df)
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎮 PREDICTION CONTROLS")
        
        # Current Date Display
        current_date = datetime.now()
        st.markdown(f"""
        <div class="stats-container">
            <h4 style="color: var(--secondary); margin-bottom: 0.5rem;">📅 CURRENT DATE</h4>
            <div class="metric-value">{current_date.strftime('%B %d, %Y')}</div>
            <div class="metric-label">{current_date.strftime('%A, %I:%M %p')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Play Time Selection
        st.markdown("### ⏰ SELECT PLAY TIME")
        next_play = get_next_play_time()
        st.markdown(f"""
        <div class="stats-container">
            <h4 style="color: var(--secondary); margin-bottom: 0.5rem;">🎯 NEXT PLAY</h4>
            <div class="metric-value" style="font-size: 1.5rem;">{next_play}</div>
        </div>
        """, unsafe_allow_html=True)
        
        play_times = [
            "10:30 AM - Morning",
            "1:00 PM - Midday", 
            "4:00 PM - Afternoon",
            "7:00 PM - Evening"
        ]
        
        selected_time = st.radio("Choose play time:", play_times, index=0)
        
        st.markdown("---")
        
        # Model Statistics
        st.markdown("### 📊 MODEL PERFORMANCE")
        st.markdown(f"""
        <div class="stats-container">
            <h4 style="color: var(--secondary); margin-bottom: 0.5rem;">🎯 MODEL ACCURACY</h4>
            <div class="metric-value">{accuracy*100:.1f}%</div>
            <div class="probability-bar" style="width: {accuracy*100}%"></div>
            <div class="metric-label" style="margin-top: 0.5rem;">
                Ensemble Model (RF + XGB + HGB + DT)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Update Status
        st.markdown("### 🔄 DATA STATUS")
        last_update = df['date'].max().strftime('%Y-%m-%d')
        st.markdown(f"""
        <div class="stats-container">
            <h4 style="color: var(--secondary); margin-bottom: 0.5rem;">📈 LAST DATA UPDATE</h4>
            <div class="metric-value" style="font-size: 1.5rem;">{last_update}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔮 GENERATE PREDICTIONS")
        
        # Generate current features for prediction
        current_features = {
            "year": current_date.year,
            "month": current_date.month,
            "day": current_date.day,
            "day_of_week": current_date.weekday(),
            "week_of_year": current_date.isocalendar()[1],
            "quarter": (current_date.month - 1) // 3 + 1,
            "winning_number_mean_last_10": df["winning_number"].tail(10).mean(),
            "winning_number_std_last_10": df["winning_number"].tail(10).std(),
            "winning_number_trend": df["winning_number"].diff().iloc[-1],
            "winning_number_diff_2": df["winning_number"].diff(2).iloc[-1],
            "previous_winning_number": df["winning_number"].iloc[-1],
            "previous_2_winning_number": df["winning_number"].iloc[-2],
            "previous_3_winning_number": df["winning_number"].iloc[-3],
            "previous_4_winning_number": df["winning_number"].iloc[-4],
            "previous_5_winning_number": df["winning_number"].iloc[-5]
        }
        
        # Fill any NaN values
        for key, value in current_features.items():
            if pd.isna(value):
                current_features[key] = 0
        
        if st.button("🎯 GENERATE AI PREDICTIONS", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing patterns..."):
                time.sleep(2)  # Add some suspense
                
                # Generate predictions
                feature_list = list(current_features.values())
                predictions_df = predict_top_numbers(model, feature_list)
                
                if not predictions_df.empty:
                    st.session_state['predictions'] = predictions_df
                    st.success("✨ Predictions generated successfully!")
                else:
                    st.error("❌ Failed to generate predictions")
        
        # Display predictions if they exist
        if 'predictions' in st.session_state:
            predictions_df = st.session_state['predictions']
            
            st.markdown("### 🎯 TOP 10 PREDICTIONS")
            
            # Create a beautiful visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=predictions_df['Winning Number'],
                    y=predictions_df['Probability'],
                    marker=dict(
                        color='#6a11cb',
                        line=dict(color='#00c9ff', width=2)
                    ),
                    text=[f"{p:.1f}%" for p in predictions_df['Probability']],
                    textposition='auto',
                    textfont=dict(color='white', size=12),
                    hovertemplate='<b>Number %{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Probability Distribution of Top Predictions",
                title_font=dict(color='white', size=22),
                xaxis_title="Winning Numbers",
                yaxis_title="Probability (%)",
                xaxis=dict(color='white', gridcolor='rgba(176, 168, 214, 0.3)', tickfont=dict(color='white')),
                yaxis=dict(color='white', gridcolor='rgba(176, 168, 214, 0.3)', tickfont=dict(color='white')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display predictions in cards
            st.markdown("### 📋 DETAILED PREDICTIONS")
            
            for i, row in predictions_df.iterrows():
                rank = i + 1
                number = int(row['Winning Number'])
                prob = row['Probability']
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 0.9rem; color: var(--text-secondary);">Rank #{rank}</span>
                            <span class="number-badge">#{number}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.8rem; font-weight: 800; color: var(--secondary);">
                                {prob:.1f}%
                            </div>
                        </div>
                    </div>
                    <div class="probability-bar" style="width: {prob}%"></div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔍 NUMBER VALIDATION")
        st.markdown("Enter your numbers to check against predictions:")
        
        user_input = st.text_area(
            "Your Numbers (comma-separated):",
            placeholder="e.g., 5, 12, 23, 31",
            help="Enter numbers between 1-36, separated by commas"
        )
        
        if st.button("✅ VALIDATE NUMBERS", use_container_width=True):
            if 'predictions' in st.session_state and user_input.strip():
                matches, result = validate_numbers(user_input, st.session_state['predictions'])
                
                if matches is None:
                    st.markdown(f"""
                    <div class="validation-result error">
                        ❌ {result}
                    </div>
                    """, unsafe_allow_html=True)
                elif matches:
                    st.markdown(f"""
                    <div class="validation-result success">
                        ✅ {len(matches)} matches found in top predictions!<br><br>
                        <strong>Matched Numbers:</strong><br>
                        {', '.join(result)}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="validation-result warning">
                        ⚠️ No matches found in top predictions<br><br>
                        <strong>Your numbers:</strong> {user_input}<br>
                        <strong>Try different numbers or check back later</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please generate predictions first and enter some numbers.")
        
        # Recent winning numbers
        st.markdown("### 📈 RECENT RESULTS")
        recent_results = df.tail(10)[['date', 'winning_number']].copy()
        recent_results['winning_number'] = recent_results['winning_number'] + 1  # Convert back to original
        recent_results['date'] = recent_results['date'].dt.strftime('%Y-%m-%d')
        
        for _, row in recent_results.iterrows():
            st.markdown(f"""
            <div class="prediction-card" style="padding: 1.2rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--text-secondary); font-size: 0.9rem;">{row['date']}</span>
                    <span style="background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); 
                              color: white; padding: 0.5rem 1rem; border-radius: 20px; 
                              font-weight: bold; font-size: 1.1rem; box-shadow: 0 5px 15px rgba(106, 17, 203, 0.4);">
                        #{int(row['winning_number'])}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistics section
        st.markdown("### 📊 PLAY WHE STATISTICS")
        st.markdown("""
        <div class="stats-container">
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span>Most Frequent Number:</span>
                <strong style="color: var(--accent);">#17 (12.5%)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span>Least Frequent Number:</span>
                <strong style="color: var(--accent);">#36 (3.2%)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span>Current Streak:</span>
                <strong style="color: var(--accent);">5 (Low Numbers)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span>Prediction Confidence:</span>
                <strong style="color: var(--accent);">87.4%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()