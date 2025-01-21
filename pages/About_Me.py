import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils.config import Config
from src.utils.styling import load_css

# Profile
col1, col2 = st.columns(spec=[0.4, 0.6], gap="medium", vertical_alignment="center")
with col1:
    st.image("img/profile_photo.png", width=210)

with col2:
    st.title("Erlangga Dwi Atha", anchor=False)
    st.subheader(
        "Data Enthusiast  |  Machine Learning Engineer."
    )
    st.write('\n')
    st.markdown('<p style="text-align: left;"><a href="https://www.linkedin.com/in/erlangga-dwi-atha/" style="color:white; text-decoration:none;">🔍 My Linkedin</a></p>', unsafe_allow_html=True)
    
# Experience
st.write("\n")
st.subheader("Experience", anchor=False)
st.write(
    """
    - 1+ years experience extracting actionable insights from data
    - Strong hands-on experience and knowledge in Python and others tools
    - Good understanding of Statistics, Computer Vision, NLP and Gen AI
    """
)

# Skills
st.write("\n")
st.subheader("Skills", anchor=False)
st.write(
    """
    - Languange: Indonesia (Native), English (Intermediate), Arabic (Basic)
    - Tools and Software:  Python, SQL, PyTorch, Tensorflow, Tableau
    - Hard Skills:  Computer Vision, NLP, Deployment of Machine Learning Models
    - Soft Skills: Problem-Solving,  Analytical Thinking, Communication, Attention to Detail
    """
)