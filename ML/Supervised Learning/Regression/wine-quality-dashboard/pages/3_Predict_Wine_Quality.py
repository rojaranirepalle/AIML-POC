from src.ui_engine import setup_ui
setup_ui()

import streamlit as st

import shap
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.model_trainer import train_models
from src.model_registry import save_model, load_model
from src.predictor import predict
from src.explainability import get_feature_importance
from src.shap_explainer import get_shap_values

st.title("🔮 Wine Prediction Engine")

df = load_data()

models, results, X = train_models(df)

best_model_name = max(results, key=lambda x: results[x]["Test"]["R2"])
best_model = models[best_model_name]

save_model(best_model)

st.success(f"Best Model: {best_model_name}")

col1, col2 = st.columns(2)

input_values = {}

for i, feature in enumerate(X.columns):

    target_col = col1 if i % 2 == 0 else col2

    with target_col:
        input_values[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=float(df[feature].mean())
        )

values = list(input_values.values())

if st.button("Predict"):

    model = load_model()

    prediction = predict(model, values)

    st.metric("Predicted Quality", round(prediction, 2))

    # Label
    if prediction >= 7:
        st.success("🍷 High Quality Wine")
    elif prediction >= 5:
        st.warning("🍷 Medium Quality Wine")
    else:
        st.error("🍷 Low Quality Wine")

    # Feature Importance
    st.subheader("📊 Feature Importance")

    importance = get_feature_importance(model, X.columns)

    if importance is not None:
        st.dataframe(importance, use_container_width=True)

    # SHAP FIXED SECTION
    st.subheader("🧠 SHAP Explanation")

    sample = df.drop("quality", axis=1).iloc[:10]

    shap_values = get_shap_values(model, sample)

    fig = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)

    st.pyplot(fig)