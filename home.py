import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.utils.config import config
from src.utils.styling import load_css

about_page = st.Page(
    "pages/About_Me.py",
    title="About Me",
    icon=":material/account_circle:",
    default=True,
)
my_projects = st.Page(
    "pages/Projects.py",
    title="My Projects",
    icon=":material/task:",
)
analytics = st.Page(
    "pages/Analytics.py",
    title="Analytics",
    icon=":material/analytics:",
)
predictions = st.Page(
    "pages/Predictions.py",
    title="Predictions",
    icon=":material/leaderboard:",
)

# Navigation
pg = st.navigation(
    {
        "Home": [about_page],
        "Projects": [my_projects, analytics, predictions],
    }
)

# Run the navigation
pg.run()