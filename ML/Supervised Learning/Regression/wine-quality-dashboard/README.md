# Wine Quality AI Dashboard

Machine Learning dashboard built with Dash (deployed on Vercel), with the original Streamlit version kept for local use.

## Features

- Data Analysis
- Multiple Regression Models
- Automatic Best Model Selection
- Model Persistence
- Explainability
- Prediction History
- Plotly Analytics
- CSV Export

## Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet

## Run

Dash (same app that runs on Vercel):

```bash
pip install -r requirements.txt
python app.py
```

Streamlit (local only):

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

## Deploy (Vercel)

```bash
vercel        # preview
vercel --prod # production
```

Vercel picks up `app.py`, which exports the Dash Flask server as `app`.
