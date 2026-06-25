import joblib
from pathlib import Path

MODEL_PATH = Path("models/best_model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)


def save_model(model):
    joblib.dump(model, MODEL_PATH)


def load_model():
    return joblib.load(MODEL_PATH)