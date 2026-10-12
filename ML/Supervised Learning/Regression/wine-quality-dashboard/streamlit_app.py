import streamlit as st
from src.ui_engine import setup_ui
setup_ui()

st.set_page_config(
    page_title="Wine AI Dashboard",
    layout="wide"
)

st.title("🍷 Wine AI Dashboard")

st.markdown("""
### Welcome

Navigate using the sidebar:

- 📊 Data Overview
- 🤖 Model Intelligence
- 🔮 Wine Quality Prediction

Features:
- EDA
- Multiple Regression Models
- Best Model Selection
- Overfitting Detection
- Prediction History
- CSV Export
""")

