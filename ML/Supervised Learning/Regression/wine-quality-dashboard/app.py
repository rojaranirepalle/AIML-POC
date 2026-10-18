from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html

from src.model_trainer import train_models
from src.utils import create_comparison_dataframe

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "winequality-red.csv"

# Chart palette (validated: blue/orange pass CVD + contrast in light and dark)
BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#898781"
GRID = "rgba(137,135,129,0.22)"
BASELINE = "rgba(137,135,129,0.55)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


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
    estimator = unwrap_pipeline(model).steps[-1][1]
    df = pd.DataFrame({"Feature": feature_names, "Impact": estimator.coef_})
    df["Abs Impact"] = df["Impact"].abs()
    return df.sort_values("Abs Impact", ascending=False).reset_index(drop=True)


def explain(model, row):
    # Linear model SHAP equivalent: intercept = average wine, coef * standardized value per feature
    pipeline = unwrap_pipeline(model)
    scaler, estimator = pipeline.steps[0][1], pipeline.steps[-1][1]
    scaled = scaler.transform(row)[0]
    contrib = pd.Series(estimator.coef_ * scaled, index=row.columns)
    return float(estimator.intercept_), contrib


def pretty(name):
    if name[1:2].isupper():  # keep "pH"
        return name
    return name[0].upper() + name[1:]


# -------------------------
# UI HELPERS
# -------------------------
def page_header(eyebrow, title, subtitle):
    return html.Header(className="page-header", children=[
        html.Div(eyebrow, className="eyebrow"),
        html.H1(title),
        html.P(subtitle, className="subtitle"),
    ])


def stat(label, value, note=None, hero=False):
    return html.Div(className="stat hero" if hero else "stat", children=[
        html.Div(label, className="stat-label"),
        html.Div(value, className="stat-value"),
        html.Div(note, className="stat-note") if note else None,
    ])


def card(title, children, subtitle=None, className=""):
    return html.Section(className=f"card {className}", children=[
        html.Div(className="card-head", children=[
            html.H2(title),
            html.P(subtitle, className="card-sub") if subtitle else None,
        ]),
        *children,
    ])


STATUS = {
    "good": ("✓", "Healthy"),
    "warning": ("!", "Watch"),
    "critical": ("✕", "Overfit"),
}


def pill(kind, label=None):
    icon, default = STATUS.get(kind, ("•", ""))
    return html.Span(className=f"pill pill-{kind}", children=[
        html.Span(icon, className="pill-icon"), label or default,
    ])


def gap_status(gap):
    if gap > 0.10:
        return "critical"
    if gap > 0.05:
        return "warning"
    return "good"


def data_table(df, page_size=10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=page_size,
        sort_action="native",
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "transparent", "color": "var(--ink-2)",
                      "fontWeight": "600", "borderBottom": "1px solid var(--grid)"},
        style_cell={"backgroundColor": "transparent", "color": "var(--ink)",
                    "border": "none", "borderBottom": "1px solid var(--grid)",
                    "padding": "8px 12px", "fontFamily": FONT, "fontSize": "13px",
                    "fontVariantNumeric": "tabular-nums", "textAlign": "right"},
        style_cell_conditional=[{"if": {"column_id": df.columns[0]}, "textAlign": "left"}],
    )


def style_fig(fig, height=320, legend=False):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=GRAY),
        margin=dict(l=8, r=16, t=8 if not legend else 36, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="#1a1a19", bordercolor="#1a1a19",
                        font=dict(family=FONT, color="#ffffff")),
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE, title=None)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE, title=None)
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, className="graph")


# -------------------------
# CHARTS
# -------------------------
def quality_distribution(df):
    counts = df["quality"].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=BLUE,
        text=counts.values, textposition="outside", cliponaxis=False,
        hovertemplate="Quality %{x}<br>%{y} wines<extra></extra>",
    ))
    fig.update_layout(bargap=0.45)
    fig.update_xaxes(dtick=1, showgrid=False)
    return style_fig(fig)


def diverging_bars(series, hover_label):
    s = series.sort_values()
    fig = go.Figure(go.Bar(
        x=s.values, y=[pretty(i) for i in s.index], orientation="h",
        marker_color=[BLUE if v >= 0 else RED for v in s.values],
        hovertemplate="%{y}<br>" + hover_label + " %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(bargap=0.35)
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, height=380)


def correlation_heatmap(df):
    corr = df.corr().round(2)
    labels = [pretty(c) for c in corr.columns]
    fig = px.imshow(
        corr.values, x=labels, y=labels, zmin=-1, zmax=1,
        color_continuous_scale=[[0, RED], [0.5, "#9a9993"], [1, BLUE]],
    )
    fig.update_traces(hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
                      xgap=2, ygap=2)
    fig.update_layout(coloraxis_colorbar=dict(thickness=10, outlinewidth=0, title=None))
    fig.update_xaxes(tickangle=-40, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, height=520)


def r2_chart(results_df):
    fig = go.Figure()
    for col, name, color in [("Train R2", "Train R²", GRAY), ("Test R2", "Test R²", BLUE)]:
        fig.add_bar(x=results_df["Model"], y=results_df[col], name=name, marker_color=color,
                    hovertemplate="%{x}<br>" + name + " %{y:.4f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.12)
    fig.update_xaxes(showgrid=False)
    return style_fig(fig, legend=True)


def rmse_chart(results_df):
    s = results_df.sort_values("Test RMSE", ascending=False)
    fig = go.Figure(go.Bar(
        x=s["Test RMSE"], y=s["Model"], orientation="h", marker_color=BLUE,
        text=s["Test RMSE"].map("{:.3f}".format), textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Test RMSE %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(bargap=0.45)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(range=[0, s["Test RMSE"].max() * 1.18])
    return style_fig(fig)


def contribution_chart(contrib):
    graph = diverging_bars(contrib, "contribution")
    # Fixed minimum span so near-zero effects look near-zero
    span = max(0.3, contrib.abs().max() * 1.15)
    graph.figure.update_xaxes(range=[-span, span])
    return graph


# -------------------------
# PAGES
# -------------------------
def home_page():
    df = load_data()
    _, results_df, X, best_name = get_models()
    best = results_df.iloc[0]
    corr_q = df.corr()["quality"].drop("quality")

    def finding(value, text):
        return html.Li([html.Div(value, className="finding-value"), html.Div(text)])

    def link_card(href, title, text, value, value_label):
        return dcc.Link(href=href, className="card link-card", children=[
            html.Div(value, className="link-value"),
            html.Div(value_label, className="stat-label"),
            html.H2(title),
            html.P(text, className="card-sub"),
            html.Span("Open →", className="link-cta"),
        ])

    return html.Div([
        page_header(
            "Red wine · regression",
            "Wine Quality Intelligence",
            "Explore the chemistry behind wine quality, compare regression models, "
            "and score a new wine in seconds.",
        ),
        html.Div(className="grid-3", children=[
            link_card("/data", "Data overview",
                      "Distribution of quality scores and which chemical properties move it.",
                      f"{len(df):,}", "wines analysed"),
            link_card("/models", "Model intelligence",
                      "Leaderboard, overfitting checks and the features that drive predictions.",
                      f"{best['Test R2']:.3f}", f"best test R² · {best_name}"),
            link_card("/predict", "Predict quality",
                      "Enter a wine's chemistry and see its predicted score explained.",
                      str(len(X.columns)), "input features"),
        ]),
        card("Key findings", [
            html.Ul(className="findings", children=[
                finding(f"{(df['quality'].between(5, 6)).mean():.0%}",
                        "of wines score 5 or 6. Truly great or poor wines are rare."),
                finding(pretty(corr_q.idxmax()),
                        f"has the strongest positive link with quality (r = {corr_q.max():.2f})."),
                finding(pretty(corr_q.idxmin()),
                        f"has the strongest negative link with quality (r = {corr_q.min():.2f})."),
                finding(f"±{best['Test MAE']:.2f}",
                        "points is the typical prediction error on unseen wines."),
            ]),
        ]),
    ])


def data_page():
    df = load_data()
    corr_q = df.corr()["quality"].drop("quality")
    high_share = (df["quality"] >= 7).mean()

    return html.Div([
        page_header("Exploration", "Data overview",
                    "Red Vinho Verde wines scored 0–10 by expert tasters."),
        html.Div(className="stats", children=[
            stat("Wines", f"{len(df):,}"),
            stat("Chemical features", df.shape[1] - 1),
            stat("Average quality", f"{df['quality'].mean():.2f}", "out of 10"),
            stat("High quality (≥ 7)", f"{high_share:.1%}", f"{(df['quality'] >= 7).sum()} wines"),
        ]),
        html.Div(className="grid-2", children=[
            card("Quality distribution", [quality_distribution(df)],
                 "Number of wines at each quality score"),
            card("What correlates with quality", [diverging_bars(corr_q, "r =")],
                 "Pearson correlation with quality · blue raises, red lowers"),
        ]),
        card("Correlation matrix", [correlation_heatmap(df)],
             "Pairwise correlation between every feature"),
        card("Dataset", [
            dcc.Tabs(className="tabs", children=[
                dcc.Tab(label="Rows", className="tab", selected_className="tab--selected",
                        children=[data_table(df.round(3))]),
                dcc.Tab(label="Summary statistics", className="tab",
                        selected_className="tab--selected",
                        children=[data_table(df.describe().round(3).reset_index()
                                             .rename(columns={"index": "stat"}))]),
            ]),
        ]),
    ])


def leaderboard(results_df):
    head = html.Tr([html.Th(h) for h in
                    ["#", "Model", "Train R²", "Test R²", "Test RMSE", "Test MAE", "Overfitting"]])
    rows = []
    for i, r in results_df.iterrows():
        rows.append(html.Tr(className="best" if i == 0 else None, children=[
            html.Td(i + 1, className="rank"),
            html.Td(r["Model"], className="name"),
            html.Td(f"{r['Train R2']:.4f}"),
            html.Td(f"{r['Test R2']:.4f}"),
            html.Td(f"{r['Test RMSE']:.4f}"),
            html.Td(f"{r['Test MAE']:.4f}"),
            html.Td([pill(gap_status(r["Overfitting Gap"])),
                     html.Span(f"{r['Overfitting Gap']:+.3f}", className="gap")]),
        ]))
    return html.Div(className="table-wrap", children=[
        html.Table(className="leaderboard", children=[html.Thead(head), html.Tbody(rows)])
    ])


def model_page():
    trained_models, results_df, X, best_name = get_models()
    best = results_df.iloc[0]
    importance = feature_importance(trained_models[best_name], X.columns)
    coefs = importance.set_index("Feature")["Impact"]
    top_pos, top_neg = coefs.idxmax(), coefs.idxmin()
    status = gap_status(best["Overfitting Gap"])

    return html.Div([
        page_header("Modelling", "Model intelligence",
                    f"Four regularised linear models trained on an 80/20 split. "
                    f"{best_name} generalises best."),
        html.Div(className="stats", children=[
            stat("Best model · test R²", f"{best['Test R2']:.3f}", best_name, hero=True),
            stat("Test RMSE", f"{best['Test RMSE']:.3f}", "quality points"),
            stat("Test MAE", f"{best['Test MAE']:.3f}", "quality points"),
            html.Div(className="stat", children=[
                html.Div("Overfitting check", className="stat-label"),
                html.Div(pill(status), className="stat-value"),
                html.Div(f"train − test R² = {best['Overfitting Gap']:+.3f}", className="stat-note"),
            ]),
        ]),
        card("Leaderboard", [leaderboard(results_df)],
             "Ranked by test R² · gap > 0.05 is watched, > 0.10 is flagged"),
        html.Div(className="grid-2", children=[
            card("Train vs test R²", [r2_chart(results_df)],
                 "A large drop from train to test signals overfitting"),
            card("Test error", [rmse_chart(results_df)], "Root mean squared error · lower is better"),
        ]),
        html.Div(className="grid-2 wide-left", children=[
            card("What drives the prediction", [diverging_bars(coefs, "coefficient")],
                 f"Standardised coefficients of {best_name} · effect of +1 std. dev."),
            card("Insights", [
                html.Ul(className="insights", children=[
                    html.Li([html.B(pretty(importance.iloc[0]["Feature"])),
                             " is the single most influential feature."]),
                    html.Li([html.B(pretty(top_pos)), f" raises quality the most "
                             f"(+{coefs[top_pos]:.3f} per std. dev.)."]),
                    html.Li([html.B(pretty(top_neg)), f" lowers quality the most "
                             f"({coefs[top_neg]:.3f} per std. dev.)."]),
                    html.Li(f"The model explains {best['Test R2']:.0%} of the variance in "
                            f"quality on unseen wines — predictions are typically off by "
                            f"about {best['Test MAE']:.2f} points."),
                ]),
            ]),
        ]),
    ])


def presets():
    df = load_data()
    features = df.drop(columns="quality")
    top = features[df["quality"] == df["quality"].max()].iloc[0]
    low = features[df["quality"] == df["quality"].min()].iloc[0]
    return {
        "preset-avg": features.mean(),
        "preset-top": top,
        "preset-low": low,
    }


PRESET_LABELS = [
    ("preset-avg", "Average wine"),
    ("preset-top", "Top-rated sample"),
    ("preset-low", "Low-rated sample"),
]


def predict_page():
    df = load_data()
    _, _, X, best_name = get_models()

    inputs = [
        html.Div(className="field", children=[
            html.Label(pretty(f), htmlFor=f"in-{i}"),
            dcc.Input(id={"type": "feature", "name": f}, type="number", step="any",
                      value=round(float(df[f].mean()), 4)),
            html.Div(f"range {df[f].min():g} – {df[f].max():g}", className="hint"),
        ])
        for i, f in enumerate(X.columns)
    ]

    return html.Div([
        page_header("Scoring", "Predict wine quality",
                    f"Scored with {best_name}. Start from a sample wine or enter your own values."),
        html.Div(className="grid-2 predict", children=[
            card("Wine chemistry", [
                html.Div(className="presets", children=[
                    html.Span("Load", className="presets-label"),
                    *[html.Button(label, id=pid, className="chip") for pid, label in PRESET_LABELS],
                ]),
                html.Div(inputs, className="fields"),
                html.Div(className="actions", children=[
                    html.Button("Predict quality", id="predict-btn", className="btn btn-primary"),
                ]),
            ]),
            dcc.Loading(html.Div(id="prediction-output"), type="dot", color=BLUE),
        ]),
    ])


PAGES = {
    "/": home_page,
    "/data": data_page,
    "/models": model_page,
    "/predict": predict_page,
}

NAV = [
    ("/", "Overview"),
    ("/data", "Data"),
    ("/models", "Models"),
    ("/predict", "Predict"),
]


# -------------------------
# APP
# -------------------------
dash_app = Dash(__name__, title="Wine Quality Intelligence",
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                assets_ignore="styles.css")  # styles.css is the Streamlit theme

dash_app.layout = html.Div(className="shell", children=[
    dcc.Location(id="url"),
    html.Nav(className="sidebar", children=[
        html.Div(className="brand", children=[
            html.Span("🍷", className="brand-mark"),
            html.Div([html.Div("Wine Quality", className="brand-name"),
                      html.Div("Intelligence", className="brand-sub")]),
        ]),
        html.Div(className="nav", children=[
            dcc.Link(label, href=href, id=f"nav-{href.strip('/') or 'home'}", className="nav-link")
            for href, label in NAV
        ]),
        html.Button("◐ Theme", id="theme-toggle", className="theme-toggle",
                    title="Toggle light / dark"),
    ]),
    html.Main(id="page", className="content"),
])


@dash_app.callback(
    Output("page", "children"),
    *[Output(f"nav-{href.strip('/') or 'home'}", "className") for href, _ in NAV],
    Input("url", "pathname"),
)
def render_page(pathname):
    page = PAGES.get(pathname)
    classes = ["nav-link active" if href == pathname else "nav-link" for href, _ in NAV]
    if page is None:
        return page_header("404", "Page not found", "Pick a page from the menu."), *classes
    return page(), *classes


@dash_app.callback(
    Output({"type": "feature", "name": ALL}, "value"),
    *[Input(pid, "n_clicks") for pid, _ in PRESET_LABELS],
    State({"type": "feature", "name": ALL}, "id"),
    prevent_initial_call=True,
)
def load_preset(*args):
    ids = args[-1]
    preset = presets()[ctx.triggered_id]
    return [round(float(preset[i["name"]]), 4) for i in ids]


@dash_app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    *[Input(pid, "n_clicks") for pid, _ in PRESET_LABELS],
    State({"type": "feature", "name": ALL}, "id"),
    State({"type": "feature", "name": ALL}, "value"),
)
def run_prediction(*args):
    ids, values = args[-2], args[-1]
    trained_models, _, X, best_name = get_models()
    model = trained_models[best_name]

    names = [i["name"] for i in ids]
    if ctx.triggered_id in dict(PRESET_LABELS):
        preset = presets()[ctx.triggered_id]
        values = [float(preset[n]) for n in names]
    elif any(v is None for v in values):
        return card("Predicted quality", [html.P("Fill in every field to get a prediction.",
                                                 className="empty")])

    row = pd.DataFrame([dict(zip(names, values))])[X.columns]
    base, contrib = explain(model, row)
    prediction = float(model.predict(row)[0])

    if prediction >= 7:
        kind, label = "good", "High quality"
    elif prediction >= 5:
        kind, label = "warning", "Medium quality"
    else:
        kind, label = "critical", "Low quality"
    icon = {"good": "▲", "warning": "●", "critical": "▼"}[kind]

    lo, hi = 3, 8
    pos = min(max((prediction - lo) / (hi - lo), 0), 1) * 100

    return html.Div(className="result-stack", children=[
        card("Predicted quality", [
            html.Div(className="result", children=[
                html.Div([html.Span(f"{prediction:.2f}", className="hero-number"),
                          html.Span(" / 10", className="hero-unit")]),
                html.Span(className=f"pill pill-{kind}",
                          children=[html.Span(icon, className="pill-icon"), label]),
            ]),
            html.Div(className="scale", children=[
                html.Div(className="scale-track", children=[
                    html.Div(className="band low"), html.Div(className="band mid"),
                    html.Div(className="band high"),
                    html.Div(className="scale-marker", style={"left": f"{pos}%"}),
                ]),
                html.Div(className="scale-labels", children=[
                    html.Span("3 · Low"), html.Span("5 · Medium"), html.Span("7 · High"),
                    html.Span("8"),
                ]),
            ]),
        ]),
        card("Why this score", [
            html.P(className="equation", children=[
                "Average wine ", html.B(f"{base:.2f}"),
                f" {'+' if prediction >= base else '−'} feature effects ",
                html.B(f"{abs(prediction - base):.2f}"), " = ", html.B(f"{prediction:.2f}"),
            ]),
            contribution_chart(contrib),
        ], "How much each feature pushed this wine's score · blue raises, red lowers"),
    ])


dash_app.clientside_callback(
    """
    function(n) {
        var root = document.documentElement, t = null;
        try { t = localStorage.getItem('wine-theme'); } catch (e) {}
        if (n) {
            var dark = t ? t === 'dark'
                         : window.matchMedia('(prefers-color-scheme: dark)').matches;
            t = dark ? 'light' : 'dark';
            try { localStorage.setItem('wine-theme', t); } catch (e) {}
        }
        if (t) { root.setAttribute('data-theme', t); }
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-toggle", "title"),
    Input("theme-toggle", "n_clicks"),
)


# Vercel looks for a top-level WSGI object named "app"
app = dash_app.server

if __name__ == "__main__":
    dash_app.run(debug=True)
