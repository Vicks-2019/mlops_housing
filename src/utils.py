
import pandas as pd


def load_csv(path):
    """Load CSV and return a DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess_data(df, features, target):
    """Simple preprocessing: fill missing values with median."""
    df = df.copy()
    df[features] = df[features].fillna(df[features].median())
    X = df[features]
    y = df[target]
    return X, y
