from src.ui_engine import setup_ui
setup_ui()

import streamlit as st

from src.data_loader import load_data
from src.visualizations import (
    plot_heatmap,
    plot_quality_distribution
)

st.title("📊 Data Overview")

df = load_data()

# Dataset Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric(
        "Average Quality",
        round(df["quality"].mean(), 2)
    )

# Dataset Preview
with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

# Summary Statistics
with st.expander("Summary Statistics"):
    st.dataframe(df.describe())

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_heatmap(df))

with col2:
    st.pyplot(plot_quality_distribution(df))