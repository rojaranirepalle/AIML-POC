import plotly.express as px


def create_r2_chart(results_df):

    fig = px.bar(
        results_df,
        x="Model",
        y="Test R2",
        color="Test R2",
        title="Model Performance Comparison"
    )

    return fig


def create_rmse_chart(results_df):

    fig = px.bar(
        results_df,
        x="Model",
        y="Test RMSE",
        color="Test RMSE",
        title="RMSE Comparison"
    )

    return fig