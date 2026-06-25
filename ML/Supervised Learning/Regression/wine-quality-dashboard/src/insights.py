def generate_model_insights(importance_df):

    top_feature = (
        importance_df.iloc[0]["Feature"]
    )

    return (
        f"The most influential feature "
        f"is '{top_feature}'."
    )