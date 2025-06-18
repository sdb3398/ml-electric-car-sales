
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import load_dataset_nations

# === STEP 1: Carica e aggiungi feature ===
df = load_dataset_nations()
df = df[df["parameter"] == "EV sales"][["nation", "year", "powertrain", "value"]]
df = df.sort_values(["nation", "powertrain", "year"])
df["sales_last_year"] = df.groupby(["nation", "powertrain"])["value"].shift(1)
df = df.dropna()

# === STEP 2: Split ===
X = df[["nation", "year", "powertrain", "sales_last_year"]]
y = df["value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 3: Encoder ===
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[["nation", "powertrain"]])

def preprocess(X):
    enc = encoder.transform(X[["nation", "powertrain"]])
    enc_df = pd.DataFrame(enc, columns=encoder.get_feature_names_out(["nation", "powertrain"]), index=X.index)
    return pd.concat([X[["year", "sales_last_year"]], enc_df], axis=1)

# === STEP 4: Cross-validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = TransformedTargetRegressor(
    regressor=XGBRegressor(random_state=42, n_estimators=100, verbosity=0),
    func=np.log1p,
    inverse_func=np.expm1
)

mae_list, rmse_list, r2_list = [], [], []

X_all = X.reset_index(drop=True)
y_all = y.reset_index(drop=True)

for train_idx, val_idx in kf.split(X_all):
    X_train_fold = preprocess(X_all.iloc[train_idx])
    X_val_fold = preprocess(X_all.iloc[val_idx])
    y_train_fold = y_all.iloc[train_idx]
    y_val_fold = y_all.iloc[val_idx]

    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_val_fold)

    mae_list.append(mean_absolute_error(y_val_fold, y_pred))
    rmse_list.append(root_mean_squared_error(y_val_fold, y_pred))
    r2_list.append(r2_score(y_val_fold, y_pred))

print("\n📊 Cross-Validation Results:")
print(f"MAE:  {np.mean(mae_list):,.2f}")
print(f"RMSE: {np.mean(rmse_list):,.2f}")
print(f"R²:   {np.mean(r2_list):.4f}")

# === STEP 5: Final training and visualization ===
model.fit(preprocess(X_train), y_train)
y_pred_final = model.predict(preprocess(X_test))

# True vs Predicted
plt.figure()
plt.scatter(y_test, y_pred_final, alpha=0.4)
plt.xlabel("True Sales")
plt.ylabel("Predicted Sales")
plt.title("True vs Predicted EV Sales")
plt.plot([0, y_test.max()], [0, y_test.max()], '--r')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/true_vs_predicted_sales_First.png")
plt.close()

# Feature importance
importances = model.regressor_.feature_importances_
features = preprocess(X_test).columns

plt.figure()
pd.Series(importances, index=features).sort_values(ascending=False).head(10).plot(kind='barh', title='Top Feature Importance')
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/feature_importance_first.png")
plt.close()