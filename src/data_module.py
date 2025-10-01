
from src.utils import load_csv, preprocess_data
from src.config import DATA_PATH, FEATURES, TARGET
from sklearn.model_selection import train_test_split


def load_and_split(test_size=0.2, random_state=42):
    """Load dataset, preprocess and split into train/test."""
    df = load_csv(DATA_PATH)
    X, y = preprocess_data(df, FEATURES, TARGET)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
