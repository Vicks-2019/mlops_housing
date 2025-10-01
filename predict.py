# predict.py
import argparse
from src.predict_module import load_model, predict
from src.config import FEATURES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", type=float, required=True,
                        help=f"Feature order: {FEATURES}")
    args = parser.parse_args()

    model = load_model()
    result = predict(model, args.features)
    print(f"Predicted House Value: ${result:,.2f}")
