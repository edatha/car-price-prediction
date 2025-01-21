import streamlit as st
import requests
import json
import pandas as pd

from src.utils.config import config

df = pd.read_csv(config.get("NEW_DATA_PATH"))

        
def run():
    st.title("🚗 Car Price Prediction")
    name = st.selectbox("Cars Model", df.name.unique())
    kms_driven = st.number_input("Kilometers driven")
    fuel_type = st.selectbox("Fuel type", df.fuel_type.unique())
    company = st.selectbox("Company Name", df.company.unique())
    car_age = st.number_input("Car Age")

    data = { 
        'name': name,
        'kms_driven': kms_driven,
        'fuel_type': fuel_type,
        'company': company,
        'car_age': car_age
        }

    if st.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        prediction = response.text
        st.success(f"The prediction from model: {prediction}")
        
if __name__ == '__main__':
    #by default it will run at 8501 port
    run()