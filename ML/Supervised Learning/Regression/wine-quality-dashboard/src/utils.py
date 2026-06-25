import pandas as pd


def create_comparison_dataframe(results):

    comparison_data = []

    for model_name in results:

        train_r2 = results[model_name]["Train"]["R2"]
        test_r2 = results[model_name]["Test"]["R2"]

        comparison_data.append({
            "Model": model_name,
            "Train R2": train_r2,
            "Test R2": test_r2,
            "Train RMSE": results[model_name]["Train"]["RMSE"],
            "Test RMSE": results[model_name]["Test"]["RMSE"],
            "Train MAE": results[model_name]["Train"]["MAE"],
            "Test MAE": results[model_name]["Test"]["MAE"],
            "Overfitting Gap": train_r2 - test_r2
        })

    return pd.DataFrame(comparison_data)