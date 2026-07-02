import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import streamlit as st 
from src.logger import logger
from src.model_registry import save_model

def train_models(df):

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=0.1),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1)
    }

    trained_models = {}
    results = {}

    for name, model in models.items():

        pipeline = make_pipeline(
            StandardScaler(),
            model
        )
        if name=="Ridge":
            grid = dict()

            grid['ridge__alpha'] = np.arange(0.1,2.1,0.1)
            search = GridSearchCV(estimator = pipeline, param_grid = grid, scoring = 'neg_mean_absolute_error',cv = 10, n_jobs= -1)

            logger.info("Training started")
            search.fit(X_train, y_train)
            #results = search.fit(X_train, y_train)
            #st.write('MAE: %.3f' % results.best_score_)
            #st.write('Config: %s' % results.best_params_)
            train_pred = search.predict(X_train)
            test_pred = search.predict(X_test)
            trained_models[name] = search
        else:   
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)

            trained_models[name] = pipeline

        results[name] = {
            "Train": {
                "R2": r2_score(y_train, train_pred),
                "RMSE": np.sqrt(mean_squared_error(y_train, train_pred)),
                "MAE": mean_absolute_error(y_train, train_pred)
            },
            "Test": {
                "R2": r2_score(y_test, test_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
                "MAE": mean_absolute_error(y_test, test_pred)
            }
        }
        logger.info("Best model selected")
    best_model_name = max(
        results,
        key=lambda x: results[x]["Test"]["R2"]
    )

    save_model(
        trained_models[best_model_name]
    )

    return trained_models, results, X

    