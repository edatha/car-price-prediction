import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils.config import Config
from src.utils.styling import load_css
from PIL import Image

st.title("🔮 My Projects")
st.write('\n')
st.write('\n')

col1, col2 = st.columns(2)
with col1: 
    cs_image = 'img/customer_segmentation.jpg'
    if cs_image is not None:
        image = Image.open(cs_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="Customer Segmentation using K-Means Clustering")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/customer-segmentation" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
    
with col2: 
    hp_image = 'img/house_price_prediction.jpg'
    if cs_image is not None:
        image = Image.open(hp_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="House Price Prediction using Random Forest Regressor")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/house-price-prediction" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
        
st.write('\n')
st.write('\n')

col1, col2 = st.columns(2)
with col1: 
    is_image = 'img/image_classification.jpg'
    if is_image is not None:
        image = Image.open(is_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="Image Classification using PyTorch")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/image-classification-pytorch" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
    
with col2: 
    cp_image = 'img/car_price_prediction.jpg'
    if cp_image is not None:
        image = Image.open(cp_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="Car Price Prediction using Linear Regression")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/car-price-prediction" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
        
st.write('\n')
st.write('\n')

col1, col2 = st.columns(2)
with col1: 
    is_image = 'img/image_classification_tensorflow.jpg'
    if is_image is not None:
        image = Image.open(is_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="Image Classification using Tensorflow")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/image-classification-tensorflow" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
    
with col2: 
    cp_image = 'img/chatbot.jpg'
    if cp_image is not None:
        image = Image.open(cp_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="Chatbot Function Call Using Google API")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/chatbot-function-calling" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
        
st.write('\n')
st.write('\n')

col1, col2 = st.columns(2)
with col1: 
    is_image = 'img/ocr.jpg'
    if is_image is not None:
        image = Image.open(is_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="OCR Using Tesseract")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/ocr-using-tesseract" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
    
with col2: 
    cp_image = 'img/rag.png'
    if cp_image is not None:
        image = Image.open(cp_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="RAG Using Haystack")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/rag-using-haystack" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1: 
    is_image = 'img/beach-litter.jpg'
    if is_image is not None:
        image = Image.open(is_image)
        new_image = image.resize((600, 350))
        st.image(new_image, caption="Beach Litter Segmentation using YOLOv8")
        st.markdown('<p style="text-align: center;"><a href="https://github.com/edatha/capstone-dibimbing" style="color:white; text-decoration:none;">Click here to visit the project</a></p>', unsafe_allow_html=True)
