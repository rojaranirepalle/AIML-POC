import pandas as pd


def get_feature_importance(model, feature_names):

    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    if hasattr(estimator, "coef_"):

        df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": estimator.coef_
        })

        df["Abs Impact"] = df["Impact"].abs()

        return df.sort_values("Abs Impact", ascending=False)

    return None