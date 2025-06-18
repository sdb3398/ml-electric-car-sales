
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
from scipy.stats.mstats import winsorize
from dataset import load_dataset_nations

# === Caricamento dati ===
df = load_dataset_nations()
df = df[df["parameter"] == "EV sales"][["nation", "year", "powertrain", "value"]]
df = df.sort_values(["nation", "powertrain", "year"])
df["sales_last_year"] = df.groupby(["nation", "powertrain"])["value"].shift(1)
df = df.dropna()

# === Winsorization del target ===
df["value_wins"] = winsorize(df["value"], limits=[0.01, 0.01])

# === Feature e target ===
X = df[["nation", "year", "powertrain", "sales_last_year"]]
y = np.log1p(df["value_wins"])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Encoding ===
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[["nation", "powertrain"]])

# === Robust Scaler su feature numeriche ===
scaler = RobustScaler()
scaler.fit(X_train[["year", "sales_last_year"]])

def preprocess(X):
    num_scaled = scaler.transform(X[["year", "sales_last_year"]])
    num_df = pd.DataFrame(num_scaled, columns=["year_scaled", "sales_last_year_scaled"], index=X.index)
    cat_enc = encoder.transform(X[["nation", "powertrain"]])
    cat_df = pd.DataFrame(cat_enc, columns=encoder.get_feature_names_out(["nation", "powertrain"]), index=X.index)
    return pd.concat([num_df, cat_df], axis=1)

# === Cross-validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = TransformedTargetRegressor(
    regressor=XGBRegressor(random_state=42, n_estimators=100, verbosity=0),
    func=np.log1p,
    inverse_func=np.expm1
)

mae_list, rmse_list, r2_list = [], [], []

X_all = X.reset_index(drop=True)
y_all = np.log1p(df["value_wins"].reset_index(drop=True))

for train_idx, val_idx in kf.split(X_all):
    X_train_fold = preprocess(X_all.iloc[train_idx])
    X_val_fold = preprocess(X_all.iloc[val_idx])
    y_train_fold = y_all.iloc[train_idx]
    y_val_fold = y_all.iloc[val_idx]

    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_val_fold)

    mae_list.append(mean_absolute_error(np.expm1(y_val_fold), np.expm1(y_pred)))
    rmse_list.append(root_mean_squared_error(np.expm1(y_val_fold), np.expm1(y_pred)))
    r2_list.append(r2_score(np.expm1(y_val_fold), np.expm1(y_pred)))

print("\n📊 Cross-Validation Results After Winsorization + Scaling:")
print(f"MAE:  {np.mean(mae_list):,.2f}")
print(f"RMSE: {np.mean(rmse_list):,.2f}")
print(f"R²:   {np.mean(r2_list):.4f}")

# === Final training and visualizations ===
model.fit(preprocess(X_train), y_train)
y_pred_final = np.expm1(model.predict(preprocess(X_test)))
y_true_final = np.expm1(y_test)

# === Scatter: True vs Predicted ===
plt.figure()
plt.scatter(y_true_final, y_pred_final, alpha=0.4)
plt.xlabel("True Sales")
plt.ylabel("Predicted Sales")
plt.title("True vs Predicted EV Sales (Outliers Winsorized)")
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
