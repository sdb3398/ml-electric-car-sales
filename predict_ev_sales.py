
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import joblib

# === Funzione per caricare il modello e la pipeline ===
def load_model_pipeline():
    model = joblib.load("xgb_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, encoder, scaler

# === Funzione per preprocessare i nuovi dati ===
def preprocess_new_data(df, encoder, scaler):
    X_cat = encoder.transform(df[["nation", "powertrain"]])
    cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(["nation", "powertrain"]))

    X_num = scaler.transform(df[["year", "sales_last_year", "growth_rate"]])
    num_df = pd.DataFrame(X_num, columns=["year_scaled", "sales_last_year_scaled", "growth_scaled"])

    return pd.concat([num_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)

# === Funzione di previsione ===
def predict_sales(input_df):
    model, encoder, scaler = load_model_pipeline()
    X_prepared = preprocess_new_data(input_df, encoder, scaler)
    preds = model.predict(X_prepared)
    return np.expm1(preds)

# === ESEMPIO USO ===
if __name__ == "__main__":
    # Esempio: previsione per il Giappone nel 2024
    df_new = pd.DataFrame([{
        "nation": "Japan",
        "year": 2024,
        "powertrain": "BEV",
        "sales_last_year": 147000,
        "growth_rate": 0.12
    }])
    pred = predict_sales(df_new)
    print(f"🔮 Predicted EV Sales: {int(pred[0]):,} vehicles")
