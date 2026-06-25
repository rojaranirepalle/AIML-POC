from src.ui_engine import setup_ui
setup_ui()

import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.model_trainer import train_models
from src.utils import create_comparison_dataframe
from src.explainability import get_feature_importance
import matplotlib.pyplot as plt
from src.insights import generate_model_insights
from src.plotly_charts import (
    create_r2_chart,
    create_rmse_chart
)

st.title("🤖 Model Intelligence")

df = load_data()

trained_models, results, X = train_models(df)

results_df = create_comparison_dataframe(results)

results_df = results_df.sort_values(
    by="Test R2",
    ascending=False
)

best_model = results_df.iloc[0]

# Leaderboard Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Test R²",
        round(best_model["Test R2"], 4)
    )

with col2:
    st.metric(
        "Best RMSE",
        round(best_model["Test RMSE"], 4)
    )

with col3:
    st.metric(
        "Best MAE",
        round(best_model["Test MAE"], 4)
    )

st.subheader("🏆 Model Leaderboard")

top3 = results_df.head(3).reset_index(drop=True)

cols = st.columns(3)

medals = ["🥇", "🥈", "🥉"]

for i, row in top3.iterrows():
    with cols[i]:
        with st.container(border=True):
            st.markdown(f"### {medals[i]} Rank {i+1}")
            st.markdown(f"**{row['Model']}**")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Train R²", f"{row['Train R2']:.4f}")
                st.metric("Train RMSE", f"{row['Train RMSE']:.4f}")

            with c2:
                st.metric("Test R²", f"{row['Test R2']:.4f}")
                st.metric("Test RMSE", f"{row['Test RMSE']:.4f}")

            gap = row["Overfitting Gap"]

            if gap > 0.10:
                st.error(f"⚠️ Overfitting Gap: {gap:.4f}")
            elif gap > 0.05:
                st.warning(f"⚠️ Overfitting Gap: {gap:.4f}")
            else:
                st.success(f"✅ Overfitting Gap: {gap:.4f}")

# Overfitting Monitor
st.subheader("⚠️ Overfitting Monitor")

results_df["Overfitting Gap"] = (
    results_df["Train R2"]
    - results_df["Test R2"]
)

st.dataframe(
    results_df[
        [
            "Model",
            "Train R2",
            "Test R2",
            "Overfitting Gap"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# Full Comparison Table
st.subheader("📊 Full Model Comparison")

st.dataframe(
    results_df.round(4),
    use_container_width=True,
    hide_index=True
)

# Feature Importance
st.subheader("🧠 Feature Importance")

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

importance_df = get_feature_importance(
    best_model,
    X.columns
)
# Feature Impact graph
if importance_df is not None:

    st.dataframe(
        importance_df,
        use_container_width=True
    )
fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(
    importance_df["Feature"],
    importance_df["Impact"]
)

ax.set_title("Feature Impact")

col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig)
with col2:
    st.subheader("📋 Model Insights")

    st.info(
        generate_model_insights(
            importance_df
        )
    )
    
st.subheader("📈 Interactive Analytics")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        create_r2_chart(results_df),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        create_rmse_chart(results_df),
        use_container_width=True
    )