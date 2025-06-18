# regression_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from dataset import load_dataset_nations

# 1. Carica e filtra
df = load_dataset_nations()
df = df[df["parameter"] == "EV sales"][["nation", "year", "powertrain", "value"]]

# 2. Feature / target
X = df.drop(columns="value")
y = df["value"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Encoder per categorie
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(X_train[["nation", "powertrain"]])

def preprocess(X):
    enc = encoder.transform(X[["nation", "powertrain"]])
    cols = encoder.get_feature_names_out(["nation", "powertrain"])
    df_enc = pd.DataFrame(enc, columns=cols, index=X.index)
    return pd.concat([X[["year"]], df_enc], axis=1)

# 5. Funzione per training e stampa
def eval_model(base_model, name):
    pipeline_X_train = preprocess(X_train)
    pipeline_X_test = preprocess(X_test)

    ttr = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    ttr.fit(pipeline_X_train, y_train)

    y_pred = ttr.predict(pipeline_X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name:20s} | MSE: {mse:,.2f} | R²: {r2:.4f}")
    return ttr

# 6. Modelli da testare
models = [
    (LinearRegression(), "Linear Regression"),
    (RandomForestRegressor(random_state=42, n_estimators=100), "Random Forest"),
    (XGBRegressor(random_state=42, n_estimators=100, verbosity=0), "XGBoost")
]

# 7. Valutazione
print("📊 Model Evaluation with log-transform target")
for model, name in models:
    eval_model(model, name)
