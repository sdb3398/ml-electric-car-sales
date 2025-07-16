# src/visualizations.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from src.preprocessing import preprocess_features, load_preprocessors

def generate_visualizations(model, encoder, scaler, X_test, y_test):
    X_processed = preprocess_features(X_test, encoder, scaler)
    y_pred = model.predict(X_processed)
    y_true = np.expm1(y_test)
    y_pred = np.expm1(y_pred)

    os.makedirs("plots", exist_ok=True)

    # Scatter plot: True vs Predicted
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.xlabel("True Sales")
    plt.ylabel("Predicted Sales")
    plt.title("True vs Predicted EV Sales")
    plt.plot([0, y_true.max()], [0, y_true.max()], "--r")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/true_vs_predicted_sales.png")
    plt.close()

    # Feature Importance
    importances = model.regressor_.feature_importances_
    features = X_processed.columns
    plt.figure()
    pd.Series(importances, index=features).sort_values(ascending=False).head(10).plot(kind="barh")
    plt.title("Top Feature Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close()
