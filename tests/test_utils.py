from src.utils import load_csv, preprocess_data
from src.config import FEATURES, TARGET


def test_preprocess_data():
    df = load_csv("housing.csv")
    X, y = preprocess_data(df, FEATURES, TARGET)
    assert not X.isnull().values.any()
    assert y.shape[0] == X.shape[0]
