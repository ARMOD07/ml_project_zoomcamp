#!/usr/bin/env python
# coding: utf-8

import pickle
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json

# -----------------------------
# 1️⃣ Load MLP model
# -----------------------------
architecture_file = "mlp_model_architecture.json"
weights_file = "mlp_model.weights.h5"
scaler_file = "mlp_model_scaler.pkl"
features_file = "feature_cols.json"

# Load model architecture
with open(architecture_file, "r") as f:
    model_json = f.read()
model = model_from_json(model_json)

# Load model weights
model.load_weights(weights_file)

# Load scaler
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# Load selected features
with open(features_file, "r") as f:
    selected_features = json.load(f)

# -----------------------------
# 2️⃣ Preprocess new data
# -----------------------------
def preprocess_new_data(df):
    """Fill missing values and scale numeric features"""
    df[selected_features] = df[selected_features].fillna(0)
    X_scaled = scaler.transform(df[selected_features])
    return X_scaled

# -----------------------------
# 3️⃣ Predict funding
# -----------------------------
def predict_funding(df_new):
    X_new = preprocess_new_data(df_new)
    preds_log = model.predict(X_new)
    preds_usd = np.expm1(preds_log)  # inverse of log1p
    return preds_usd

# -----------------------------
# 4️⃣ Example: Predict funding
# -----------------------------
if __name__ == "__main__":
    json_data = [
        {
            "nb_funding_rounds": 3,
            "nb_investors": 10,
            "nb_offices": 2,
            "ipo": 0,
            "acquired": 0,
            "milestones": 5,
            "relationships": 7,
            "funding_rounds": 3
        }
    ]

    df_new = pd.DataFrame(json_data)
    predictions = predict_funding(df_new)

    print("Predicted funding (USD):", predictions)