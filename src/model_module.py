
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from src.config import MODEL_PATH


def train_model(X_train, y_train):
    """Train linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, R²: {r2:.4f}")
    return y_pred


def save_model(model):
    """Save trained model to disk."""
    joblib.dump(model, MODEL_PATH)
