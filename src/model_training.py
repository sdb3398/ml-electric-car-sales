# src/model_training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score, root_mean_squared_error,
    mean_absolute_percentage_error, median_absolute_error, max_error
)
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from src.preprocessing import preprocess_features, fit_preprocessors
from scipy.stats.mstats import winsorize
import joblib


def train_and_evaluate_model(df: pd.DataFrame):
    # === Feature engineering ===
    df = df[df["parameter"] == "EV sales"][["nation", "year", "powertrain", "value"]]
    df = df.sort_values(["nation", "powertrain", "year"])
    df["sales_last_year"] = df.groupby(["nation", "powertrain"])["value"].shift(1)
    df["growth_rate"] = (df["value"] - df["sales_last_year"]) / df["sales_last_year"]
    df = df.dropna()

    # === Winsorization ===
    df["value_wins"] = winsorize(df["value"], limits=[0.01, 0.01])

    # === Define X and y ===
    X = df[["nation", "year", "powertrain", "sales_last_year", "growth_rate"]]
    y = np.log1p(df["value_wins"])

    # === Fit encoder and scaler ===
    encoder, scaler = fit_preprocessors(X)

    # === Prepare features ===
    X_prepared = preprocess_features(X, encoder, scaler)

    # === Setup model and CV ===
    model = TransformedTargetRegressor(
        regressor=XGBRegressor(random_state=42, n_estimators=100, verbosity=0),
        func=np.log1p,
        inverse_func=np.expm1
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {
        "MAE": [], "RMSE": [], "R2": [], "MAPE": [], "MedAE": [], "MaxError": []
    }

    for train_idx, val_idx in kf.split(X_prepared):
        X_train_cv, X_val_cv = X_prepared.iloc[train_idx], X_prepared.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_val_cv)

        y_true_exp = np.expm1(y_val_cv)
        y_pred_exp = np.expm1(y_pred)

        metrics["MAE"].append(mean_absolute_error(y_true_exp, y_pred_exp))
        metrics["RMSE"].append(root_mean_squared_error(y_true_exp, y_pred_exp))
        metrics["R2"].append(r2_score(y_true_exp, y_pred_exp))
        metrics["MAPE"].append(mean_absolute_percentage_error(y_true_exp, y_pred_exp))
        metrics["MedAE"].append(median_absolute_error(y_true_exp, y_pred_exp))
        metrics["MaxError"].append(max_error(y_true_exp, y_pred_exp))

    print("\n\U0001F4CA Cross-Validation Metrics with Growth Rate and Extended Evaluation:")
    for m, scores in metrics.items():
        print(f"{m:6s}: {np.mean(scores):,.2f}")

    # === Final fit ===
    model.fit(X_prepared, y)

    # === Save ===
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(encoder, "models/encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    return model, encoder, scaler
