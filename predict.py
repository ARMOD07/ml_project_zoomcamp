#!/usr/bin/env python
# coding: utf-8

import pickle
import json  # <-- il manquait ça
import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json

# -----------------------------
# 1️⃣ Charger le modèle MLP
# -----------------------------
architecture_file = "Projects/mlp_model_architecture.json"
weights_file = "Projects/mlp_model.weights.h5"
scaler_file = "Projects/mlp_model_scaler.pkl"
features_file = "Projects/feature_cols.json"

# Charger l'architecture
with open(architecture_file, "r") as f:
    model_json = f.read()

model = model_from_json(model_json)

# Charger les poids
model.load_weights(weights_file)

# Charger le scaler
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

# Charger les features sélectionnées
with open(features_file, "r") as f:
    selected_features = json.load(f)

# -----------------------------
# 2️⃣ Prétraitement des nouvelles données
# -----------------------------
def preprocess_new_data(df):
    df[selected_features] = df[selected_features].fillna(0)
    X_scaled = scaler.transform(df[selected_features])
    return X_scaled

# -----------------------------
# 3️⃣ Prédiction
# -----------------------------
def predict_funding(df_new):
    X_new = preprocess_new_data(df_new)
    preds = model.predict(X_new)
    return preds

# -----------------------------
# 4️⃣ Exemple JSON à prédire
# -----------------------------
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

print("Predicted funding:", predictions)
pred_original = np.expm1(predictions)  # inverse de np.log1p
print("Predicted funding (USD):", pred_original)
