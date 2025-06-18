import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib


def fit_preprocessors(X: pd.DataFrame):
    """
    Fit encoder and scaler on input features.
    """
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(X[["nation", "powertrain"]])

    scaler = RobustScaler()
    scaler.fit(X[["year", "sales_last_year", "growth_rate"]])

    return encoder, scaler


def preprocess_features(X: pd.DataFrame, encoder: OneHotEncoder, scaler: RobustScaler) -> pd.DataFrame:
    """
    Apply preprocessing: encoding + scaling.
    """
    X_num = scaler.transform(X[["year", "sales_last_year", "growth_rate"]])
    df_num = pd.DataFrame(X_num, columns=["year_scaled", "sales_last_year_scaled", "growth_scaled"], index=X.index)

    X_cat = encoder.transform(X[["nation", "powertrain"]])
    df_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(["nation", "powertrain"]), index=X.index)

    return pd.concat([df_num, df_cat], axis=1)


def save_preprocessors(encoder: OneHotEncoder, scaler: RobustScaler):
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")


def load_preprocessors():
    encoder = joblib.load("models/encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return encoder, scaler
