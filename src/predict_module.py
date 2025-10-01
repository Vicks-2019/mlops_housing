
import joblib
import pandas as pd
from src.config import MODEL_PATH, FEATURES

def load_model():
    return joblib.load(MODEL_PATH)

def predict(model, feature_values):
    X = pd.DataFrame([feature_values], columns=FEATURES)
    return model.predict(X)[0]
