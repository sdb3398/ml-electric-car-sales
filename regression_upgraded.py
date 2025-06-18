
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (mean_absolute_error, r2_score, root_mean_squared_error,
                             mean_absolute_percentage_error, median_absolute_error, max_error)
from xgboost import XGBRegressor
from scipy.stats.mstats import winsorize
import joblib
from dataset import load_dataset_nations

# === Load and feature engineering ===
df = load_dataset_nations()
df = df[df["parameter"] == "EV sales"][["nation", "year", "powertrain", "value"]]
df = df.sort_values(["nation", "powertrain", "year"])
df["sales_last_year"] = df.groupby(["nation", "powertrain"])["value"].shift(1)
df["growth_rate"] = (df["value"] - df["sales_last_year"]) / df["sales_last_year"]
df = df.dropna()

# === Winsorization of target ===
df["value_wins"] = winsorize(df["value"], limits=[0.01, 0.01])

# === Feature and target ===
X = df[["nation", "year", "powertrain", "sales_last_year", "growth_rate"]]
y = np.log1p(df["value_wins"])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Encoding ===
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[["nation", "powertrain"]])

# === Robust scaling ===
scaler = RobustScaler()
scaler.fit(X_train[["year", "sales_last_year", "growth_rate"]])

def preprocess(X):
    X_num = scaler.transform(X[["year", "sales_last_year", "growth_rate"]])
    df_num = pd.DataFrame(X_num, columns=["year_scaled", "sales_last_year_scaled", "growth_scaled"], index=X.index)
    X_cat = encoder.transform(X[["nation", "powertrain"]])
    df_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(["nation", "powertrain"]), index=X.index)
    return pd.concat([df_num, df_cat], axis=1)

# === Model setup ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = TransformedTargetRegressor(
    regressor=XGBRegressor(random_state=42, n_estimators=100, verbosity=0),
    func=np.log1p,
    inverse_func=np.expm1
)

# === Cross-validation metrics ===
metrics = {
    "MAE": [],
    "RMSE": [],
    "R2": [],
    "MAPE": [],
    "MedAE": [],
    "MaxError": []
}

X_all = X.reset_index(drop=True)
y_all = np.log1p(df["value_wins"].reset_index(drop=True))

for train_idx, val_idx in kf.split(X_all):
    X_train_fold = preprocess(X_all.iloc[train_idx])
    X_val_fold = preprocess(X_all.iloc[val_idx])
    y_train_fold = y_all.iloc[train_idx]
    y_val_fold = y_all.iloc[val_idx]

    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_val_fold)

    y_true_exp = np.expm1(y_val_fold)
    y_pred_exp = np.expm1(y_pred)

    metrics["MAE"].append(mean_absolute_error(y_true_exp, y_pred_exp))
    metrics["RMSE"].append(root_mean_squared_error(y_true_exp, y_pred_exp))
    metrics["R2"].append(r2_score(y_true_exp, y_pred_exp))
    metrics["MAPE"].append(mean_absolute_percentage_error(y_true_exp, y_pred_exp))
    metrics["MedAE"].append(median_absolute_error(y_true_exp, y_pred_exp))
    metrics["MaxError"].append(max_error(y_true_exp, y_pred_exp))

# === Print results ===
print("\n📊 Cross-Validation Metrics with Growth Rate and Extended Evaluation:")
for m, scores in metrics.items():
    print(f"{m:6s}: {np.mean(scores):,.2f}")

# === Final training and visualizations ===
model.fit(preprocess(X_train), y_train)
y_pred_final = np.expm1(model.predict(preprocess(X_test)))
y_true_final = np.expm1(y_test)

# === Scatter plot ===
plt.figure()
plt.scatter(y_true_final, y_pred_final, alpha=0.4)
plt.xlabel("True Sales")
plt.ylabel("Predicted Sales")
plt.title("True vs Predicted EV Sales (Extended Model)")
plt.plot([0, y_true_final.max()], [0, y_true_final.max()], '--r')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Feature importance ===
importances = model.regressor_.feature_importances_
features = preprocess(X_test).columns

plt.figure()
pd.Series(importances, index=features).sort_values(ascending=False).head(10).plot(kind='barh', title='Top Feature Importance')
plt.grid(True)
plt.tight_layout()
plt.show()


# Salvataggio dei modelli e dei trasformatori
joblib.dump(model, "xgb_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")


