import streamlit as st
import requests
import pandas as pd

from src.utils.config import config

# Load data
df = pd.read_csv(config.get("NEW_DATA_PATH"))

# Set backend URL
BACKEND_URL = config.get("BACKEND_URL", "http://127.0.0.1:8000")

# Title
st.title("🚗 Car Price Prediction")

# Input fields
name = st.selectbox("Cars Model", df.name.unique())
kms_driven = st.number_input("Kilometers driven")
fuel_type = st.selectbox("Fuel type", df.fuel_type.unique())
company = st.selectbox("Company Name", df.company.unique())
car_age = st.number_input("Car Age")

# Prepare data
data = {
    'name': name,
    'kms_driven': kms_driven,
    'fuel_type': fuel_type,
    'company': company,
    'car_age': car_age
}

# Prediction
if st.button("Predict"):
    with st.spinner("Fetching prediction..."):
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=data)
            response.raise_for_status()
            prediction = response.json().get("predicted_price", "Error: Invalid Response")
            st.success(f"The prediction from model: {prediction}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error contacting API: {e}")