# src/config.py

DATA_PATH = "housing.csv"
MODEL_PATH = "model.joblib"

FEATURES = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]
TARGET = "median_house_value"
