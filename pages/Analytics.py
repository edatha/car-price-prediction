import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.utils.config import config
import json

st.title("📊 Data Analytics")
# st.divider()

@st.cache_data(ttl=3600)
def load_data():
    try:
        return pd.read_csv(config.get('NEW_DATA_PATH'))
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

        
        
df = load_data()
metrics= None

with open(config.get('METRICS_PATH'), 'r') as f:
    metrics = json.load(f)

features = config.get('FEATURE_IMPORTANCE')

tab1, tab2, tab3 = st.tabs([
    "Reading Data",
    "Feature Relationship (Numerical Column)",
    "Model Performance"
])

with tab1:
    st.header("Reading Our Data - Car Pricing")
    st.write('\n')
    st.write('\n')
    col1, col2, col3 = st.columns(3)
    
    number_of_row = df.shape[0]
    mean_of_price = df[config.get('TARGET_COLUMN')].mean().round(2)
    number_of_col = df.shape[1]
    
    with col1:
        st.metric("Number of Records: ", number_of_row)
    with col2:
        st.metric("Average Car Price : ", f"${mean_of_price}")
    with col3:
        st.metric("Features Used : ", number_of_col)
    
    st.markdown("### 🔍 Sample Data")
    st.dataframe(df.head(20))
    
with tab2:
    st.header("Numerical Col x Target")
    st.write('\n')
    st.write('\n')
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis feature", config.get('NUMERICAL_COL'))
    with col2:
        y_feature = st.selectbox(
            "Select Y-axis feature", 
            [config.get('TARGET_COLUMN')], 
            index=1 if len([config.get('TARGET_COLUMN')]) > 1 else 0)
    
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        title=f"{x_feature} vs {y_feature}",
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)
    
with tab3:
    st.header("Model Performance in 3 Matrices")
    st.write('\n')
    st.write('\n')
    
    if metrics and features:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
        with col2:
            st.metric("Test RMSE", f"{metrics['test_rmse']:,.2f}")
        with col3:
            st.metric("Test MAE", f"{metrics['test_mae']:,.2f}")
            
        st.subheader("Training vs Testing Performance")
        metrics_df = pd.DataFrame({
            'Metric': ['R²', 'RMSE', 'MAE'],
            'Training': [
                metrics['train_r2'],
                metrics['train_rmse'],
                metrics['train_mae']
            ],
            'Testing': [
                metrics['test_r2'],
                metrics['test_rmse'],
                metrics['test_mae']
            ]
        })
        st.dataframe(metrics_df)