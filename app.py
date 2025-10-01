
from src.data_module import load_and_split
from src.model_module import train_model, evaluate_model, save_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)