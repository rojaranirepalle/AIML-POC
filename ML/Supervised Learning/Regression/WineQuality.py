import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# -------------------------
# Netflix-style UI config
# -------------------------
st.set_page_config(
    page_title="Wine AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stMetric {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        border: 2px solid black;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍷 Wine AI Intelligence Dashboard")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("wine_dataset/winequality-red.csv", sep=";")

df = load_data()

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("⚙️ Controls")

show_data = st.sidebar.checkbox("Show Dataset")

# -------------------------
# Data Overview Cards
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Avg Quality", round(df["quality"].mean(), 2))

if show_data:
    st.dataframe(df)

# -------------------------
# EDA SECTION
# -------------------------
st.subheader("📊 Data Intelligence Layer")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x="quality", data=df, ax=ax2)
    st.pyplot(fig2)

# -------------------------
# Train Models
# -------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1)
}

trained_models = {}
results = {}

for name, model in models.items():
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)

    y_test_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    trained_models[name] = pipeline

    results[name] = {
        "Train": {
            "MAE": mean_absolute_error(y_train, y_train_pred),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "R2": r2_score(y_train, y_train_pred)
        },
        "Test": {
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "R2": r2_score(y_test, y_test_pred)
        }
    }

# -------------------------
# BEST MODEL SELECTION (FIXED)
# -------------------------

flattened_results = {}

for model_name in results:
    train_r2 = results[model_name]["Train"]["R2"]
    test_r2 = results[model_name]["Test"]["R2"]
    flattened_results[model_name] = {
        "Train R2": results[model_name]["Train"]["R2"],
        "Test R2": results[model_name]["Test"]["R2"],
        "Train RMSE": results[model_name]["Train"]["RMSE"],
        "Test RMSE": results[model_name]["Test"]["RMSE"],
        "Train MAE": results[model_name]["Train"]["MAE"],
        "Test MAE": results[model_name]["Test"]["MAE"],
        "Overfitting Gap": train_r2 - test_r2
    }

results_df = pd.DataFrame.from_dict(flattened_results, orient="index")

results_df = results_df.sort_values(by="Test R2", ascending=False).reset_index()
results_df = results_df.rename(columns={"index": "Model"})

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]

st.subheader("🤖 Model Intelligence Table")

def highlight(row):
    if row["Model"] == best_model_name:
        return ["background-color: #2ecc71; color: black"] * len(row)

    # 🔥 highlight overfitting risk
    if row["Overfitting Gap"] > 0.10:
        return ["background-color: #ffcccb"] * len(row)

    return [""] * len(row)

st.dataframe(
    results_df.style.apply(highlight, axis=1),
    use_container_width=True,
    hide_index=True
)

# st.success(f"🏆 Best Model: {best_model_name}")

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

# -------------------------
# SIDE-BY-SIDE DASHBOARD SECTION
# -------------------------


col1, col2 = st.columns(2)

# -------------------------
# LEFT: Model Performance Radar
# -------------------------
with col1:
    st.markdown("### 📊 Model Performance Radar")

    fig, ax = plt.subplots()
    sns.barplot(data=results_df, x="Model", y="Test R2", ax=ax)
    ax.set_ylabel("R² Score")
    ax.set_title("Model Comparison (Test R²)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------
# RIGHT: Explainability
# -------------------------
with col2:
    st.markdown("### 🧠 AI Explainability Layer")

    model_for_explain = best_model.named_steps[
        list(best_model.named_steps.keys())[-1]
    ]

    if hasattr(model_for_explain, "coef_"):
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Impact": model_for_explain.coef_
        }).sort_values(by="Impact")

        fig2, ax2 = plt.subplots()
        ax2.barh(importance["Feature"], importance["Impact"])
        ax2.set_title("Feature Impact (Coefficients)")
        st.pyplot(fig2)
    else:
        st.info("Explainability not available for this model.")

# -------------------------
# PREDICTION ENGINE
# -------------------------
st.subheader("🔮 Wine Quality Predictor")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
    citric_acid = st.number_input("Citric Acid", value=0.0)
    residual_sugar = st.number_input("Residual Sugar", value=2.0)
    chlorides = st.number_input("Chlorides", value=0.08)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=46.0)
    density = st.number_input("Density", value=0.996)
    pH = st.number_input("pH", value=3.3)
    sulphates = st.number_input("Sulphates", value=0.6)
    alcohol = st.number_input("Alcohol", value=10.0)

if st.button("Predict Quality"):
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]])

    prediction = best_model.predict(input_data)

    st.metric("Predicted Wine Quality", round(prediction[0], 2))
