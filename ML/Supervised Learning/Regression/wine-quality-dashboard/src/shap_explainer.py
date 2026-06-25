import shap
import numpy as np


def get_shap_values(model, X_sample):

    # Extract model safely from pipeline
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    X_sample_np = X_sample.values

    # SAFE explainer for linear models
    explainer = shap.LinearExplainer(estimator, X_sample_np)

    shap_values = explainer(X_sample_np)

    return shap_values