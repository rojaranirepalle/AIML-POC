import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(df):

    fig, ax = plt.subplots()

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    return fig


def plot_quality_distribution(df):

    fig, ax = plt.subplots()

    sns.countplot(
        x="quality",
        data=df,
        ax=ax
    )

    return fig


def plot_model_comparison(results_df):

    fig, ax = plt.subplots()

    sns.barplot(
        data=results_df,
        x="Model",
        y="Test R2",
        ax=ax
    )

    plt.xticks(rotation=45)

    return fig