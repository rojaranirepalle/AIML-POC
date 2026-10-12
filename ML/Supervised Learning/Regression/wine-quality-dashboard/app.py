from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import ALL, Dash, Input, Output, State, dash_table, dcc, html

from src.insights import generate_model_insights
from src.model_trainer import train_models
from src.plotly_charts import create_r2_chart, create_rmse_chart
from src.utils import create_comparison_dataframe

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "winequality-red.csv"


# -------------------------
# DATA + MODELS (cached per server instance)
# -------------------------
@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")


@lru_cache(maxsize=1)
def get_models():
    df = load_data()
    trained_models, results, X = train_models(df)
    results_df = create_comparison_dataframe(results).sort_values(
        by="Test R2", ascending=False
    ).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]
    return trained_models, results_df, X, best_name


def unwrap_pipeline(model):
    # GridSearchCV wraps the pipeline (Ridge)
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    return model


def feature_importance(model, feature_names):
    pipeline = unwrap_pipeline(model)
    estimator = pipeline.steps[-1][1]
    df = pd.DataFrame({"Feature": feature_names, "Impact": estimator.coef_})
    df["Abs Impact"] = df["Impact"].abs()
    return df.sort_values("Abs Impact", ascending=False).reset_index(drop=True)


def contributions(model, row):
    # Linear model SHAP equivalent: coef * standardized value
    pipeline = unwrap_pipeline(model)
    scaler, estimator = pipeline.steps[0][1], pipeline.steps[-1][1]
    scaled = scaler.transform(row)[0]
    df = pd.DataFrame({
        "Feature": row.columns,
        "Contribution": estimator.coef_ * scaled
    })
    return df.reindex(df["Contribution"].abs().sort_values().index)


# -------------------------
# UI HELPERS
# -------------------------
def metric(label, value):
    return html.Div(className="metric", children=[
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ])


def table(df, page_size=10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=page_size,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1c1f26", "color": "#fff", "fontWeight": "600"},
        style_cell={"backgroundColor": "#161a22", "color": "#e6e6e6",
                    "border": "1px solid #2a2f3a", "padding": "6px"},
    )


def graph(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=20))
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def alert(text, kind):
    return html.Div(text, className=f"alert alert-{kind}")


def gap_alert(gap):
    if gap > 0.10:
        return alert(f"⚠️ Overfitting Gap: {gap:.4f}", "error")
    if gap > 0.05:
        return alert(f"⚠️ Overfitting Gap: {gap:.4f}", "warning")
    return alert(f"✅ Overfitting Gap: {gap:.4f}", "success")


# -------------------------
# PAGES
# -------------------------
def home_page():
    return html.Div([
        html.H1("🍷 Wine AI Dashboard"),
        html.H3("Welcome"),
        html.P("Navigate using the sidebar:"),
        html.Ul([
            html.Li("📊 Data Overview"),
            html.Li("🤖 Model Intelligence"),
            html.Li("🔮 Wine Quality Prediction"),
        ]),
        html.P("Features:"),
        html.Ul([html.Li(f) for f in [
            "EDA", "Multiple Regression Models", "Best Model Selection",
            "Overfitting Detection", "Linear Explainability",
        ]]),
    ])


def data_page():
    df = load_data()

    heatmap = px.imshow(df.corr().round(2), color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Correlation Heatmap")
    dist = px.histogram(df, x="quality", title="Quality Distribution")
    dist.update_layout(bargap=0.1)

    return html.Div([
        html.H1("📊 Data Overview"),
        html.Div(className="row", children=[
            metric("Rows", df.shape[0]),
            metric("Columns", df.shape[1]),
            metric("Average Quality", round(df["quality"].mean(), 2)),
        ]),
        html.Details([html.Summary("View Dataset"), table(df)]),
        html.Details([html.Summary("Summary Statistics"),
                      table(df.describe().round(3).reset_index())]),
        html.Div(className="row", children=[
            html.Div(graph(heatmap), className="col"),
            html.Div(graph(dist), className="col"),
        ]),
    ])


def model_page():
    trained_models, results_df, X, best_name = get_models()
    best = results_df.iloc[0]
    importance_df = feature_importance(trained_models[best_name], X.columns)

    medals = ["🥇", "🥈", "🥉"]
    cards = []
    for i, row in results_df.head(3).iterrows():
        cards.append(html.Div(className="card col", children=[
            html.H3(f"{medals[i]} Rank {i + 1}"),
            html.P(html.B(row["Model"])),
            html.Div(className="row", children=[
                metric("Train R²", f"{row['Train R2']:.4f}"),
                metric("Test R²", f"{row['Test R2']:.4f}"),
            ]),
            html.Div(className="row", children=[
                metric("Train RMSE", f"{row['Train RMSE']:.4f}"),
                metric("Test RMSE", f"{row['Test RMSE']:.4f}"),
            ]),
            gap_alert(row["Overfitting Gap"]),
        ]))

    impact_fig = px.bar(importance_df.iloc[::-1], x="Impact", y="Feature",
                        orientation="h", title="Feature Impact")

    return html.Div([
        html.H1("🤖 Model Intelligence"),
        html.Div(className="row", children=[
            metric("Best Test R²", round(best["Test R2"], 4)),
            metric("Best RMSE", round(best["Test RMSE"], 4)),
            metric("Best MAE", round(best["Test MAE"], 4)),
        ]),
        html.H2("🏆 Model Leaderboard"),
        html.Div(className="row", children=cards),
        html.H2("⚠️ Overfitting Monitor"),
        table(results_df[["Model", "Train R2", "Test R2", "Overfitting Gap"]].round(4)),
        html.H2("📊 Full Model Comparison"),
        table(results_df.round(4)),
        html.H2("🧠 Feature Importance"),
        table(importance_df.round(4), page_size=12),
        html.Div(className="row", children=[
            html.Div(graph(impact_fig), className="col"),
            html.Div(className="col", children=[
                html.H3("📋 Model Insights"),
                alert(generate_model_insights(importance_df), "info"),
            ]),
        ]),
        html.H2("📈 Interactive Analytics"),
        html.Div(className="row", children=[
            html.Div(graph(create_r2_chart(results_df)), className="col"),
            html.Div(graph(create_rmse_chart(results_df)), className="col"),
        ]),
    ])


def predict_page():
    df = load_data()
    _, _, X, best_name = get_models()

    inputs = [
        html.Div(className="input", children=[
            html.Label(feature.replace("_", " ").title()),
            dcc.Input(id={"type": "feature", "name": feature}, type="number",
                      value=round(float(df[feature].mean()), 4), step="any"),
        ])
        for feature in X.columns
    ]

    return html.Div([
        html.H1("🔮 Wine Prediction Engine"),
        alert(f"Best Model: {best_name}", "success"),
        html.Div(inputs, className="input-grid"),
        html.Button("Predict", id="predict-btn", className="btn"),
        html.Div(id="prediction-output"),
    ])


PAGES = {
    "/": home_page,
    "/data": data_page,
    "/models": model_page,
    "/predict": predict_page,
}

NAV = [
    ("/", "🏠 Home"),
    ("/data", "📊 Data Overview"),
    ("/models", "🤖 Model Intelligence"),
    ("/predict", "🔮 Predict Wine Quality"),
]


# -------------------------
# APP
# -------------------------
dash_app = Dash(__name__, title="Wine AI Dashboard", suppress_callback_exceptions=True)

dash_app.layout = html.Div(className="layout", children=[
    dcc.Location(id="url"),
    html.Nav(className="sidebar", children=[
        html.Div("🍷 Wine AI", className="brand"),
        *[dcc.Link(label, href=href, className="nav-link") for href, label in NAV],
    ]),
    html.Main(id="page", className="content"),
])


@dash_app.callback(Output("page", "children"), Input("url", "pathname"))
def render_page(pathname):
    page = PAGES.get(pathname)
    if page is None:
        return html.H1("404 — Page not found")
    return page()


@dash_app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State({"type": "feature", "name": ALL}, "id"),
    State({"type": "feature", "name": ALL}, "value"),
    prevent_initial_call=True,
)
def run_prediction(_, ids, values):
    trained_models, _, X, best_name = get_models()
    model = trained_models[best_name]

    if any(v is None for v in values):
        return alert("Please fill in every feature.", "error")

    row = pd.DataFrame([dict(zip([i["name"] for i in ids], values))])[X.columns]
    prediction = float(model.predict(row)[0])

    if prediction >= 7:
        label = alert("🍷 High Quality Wine", "success")
    elif prediction >= 5:
        label = alert("🍷 Medium Quality Wine", "warning")
    else:
        label = alert("🍷 Low Quality Wine", "error")

    contrib = contributions(model, row)
    contrib_fig = px.bar(contrib, x="Contribution", y="Feature", orientation="h",
                         color="Contribution", color_continuous_scale="RdBu",
                         title="Contribution to this prediction (vs. average wine)")

    return html.Div([
        html.Div(className="row", children=[metric("Predicted Quality", round(prediction, 2))]),
        label,
        html.H2("📊 Feature Importance"),
        table(feature_importance(model, X.columns).round(4), page_size=12),
        html.H2("🧠 Prediction Explanation"),
        graph(contrib_fig),
    ])


# Vercel looks for a top-level WSGI object named "app"
app = dash_app.server

if __name__ == "__main__":
    dash_app.run(debug=True)
