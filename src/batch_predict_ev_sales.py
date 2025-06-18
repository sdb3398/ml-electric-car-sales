
import pandas as pd
import numpy as np
import joblib
from predict_ev_sales import preprocess_new_data
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import load_dataset_nations

# === Caricamento modelli salvati ===
model = joblib.load("xgb_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# === Caricamento dati originali ===
df = load_dataset_nations()
df = df[df["parameter"] == "EV sales"].copy()
df = df.sort_values(["nation", "powertrain", "year"])

# === Calcolo sales_last_year e growth ===
df["sales_last_year"] = df.groupby(["nation", "powertrain"])["value"].shift(1)
df["growth_rate"] = df.groupby(["nation", "powertrain"])["value"].pct_change()

# === Estrazione dati del 2023 per costruire quelli del 2024 ===
df_2023 = df[df["year"] == 2023].dropna(subset=["sales_last_year", "growth_rate"]).copy()
df_2024 = df_2023.copy()
df_2024["year"] = 2024
df_2024["sales_last_year"] = df_2023["value"]
df_2024["growth_rate"] = df_2023["growth_rate"]

# === Input per la predizione
X_2024 = df_2024[["nation", "year", "powertrain", "sales_last_year", "growth_rate"]].copy()
X_prepared = preprocess_new_data(X_2024, encoder, scaler)

# === Predizione
y_pred = model.predict(X_prepared)
df_2024["predicted_sales"] = np.expm1(y_pred).round().astype(int)

# === Salvataggio
df_2024[["nation", "powertrain", "year", "sales_last_year", "growth_rate", "predicted_sales"]]\
    .sort_values("predicted_sales", ascending=False)\
    .to_csv("data/EV_sales_predictions_2024.csv", index=False)

print("✅ Previsioni 2024 generate con base 2023. File salvato: EV_sales_predictions_2024.csv")
