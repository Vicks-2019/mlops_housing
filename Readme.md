# Simple MLOps Linear Regression


This minimal project demonstrates a **3-module** setup for linear regression.


Modules:
- `data_module.py` – load & clean data
- `model_module.py` – build & train model
- `predict_module.py` – predict & test


### Steps
1. Install requirements: `pip install -r requirements.txt`
2. Train: `python train.py`
3. Predict: `python predict.py --features 1.2 3.4 5.6`