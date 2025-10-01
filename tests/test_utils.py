# tests/test_utils.py
from src.utils import preprocess_data
import pandas as pd

def test_preprocess_data():
    df = pd.DataFrame({
        "longitude": [1, None],
        "latitude": [2, 3],
        "housing_median_age": [10, 20],
        "total_rooms": [100, 200],
        "total_bedrooms": [50, None],
        "population": [300, 400],
        "households": [100, 150],
        "median_income": [2.5, 3.0],
        "median_house_value": [100000, 200000]
    })
    X, y = preprocess_data(df,
                           ["longitude","latitude","housing_median_age",
                            "total_rooms","total_bedrooms","population",
                            "households","median_income"],
                           "median_house_value")
    assert X.isnull().sum().sum() == 0
    assert len(y) == 2
