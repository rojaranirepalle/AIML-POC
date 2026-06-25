import pandas as pd


def compute_score(train_r2, test_r2, rmse):

    return (
        0.5 * test_r2 +
        0.3 * (1 - rmse) +
        0.2 * (1 - abs(train_r2 - test_r2))
    )


def build_leaderboard(results):

    rows = []

    for model in results:

        tr = results[model]["Train"]
        te = results[model]["Test"]

        rows.append({
            "Model": model,
            "Train R2": tr["R2"],
            "Test R2": te["R2"],
            "Test RMSE": te["RMSE"],
            "Overfitting Gap": tr["R2"] - te["R2"],
            "Score": compute_score(tr["R2"], te["R2"], te["RMSE"])
        })

    df = pd.DataFrame(rows)

    return df.sort_values("Score", ascending=False)