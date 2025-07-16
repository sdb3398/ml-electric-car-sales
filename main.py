from dataset import load_dataset_nations
from src.model_training import train_and_evaluate_model
from src.visualizations import generate_visualizations
import joblib
import os

# === Load dataset ===
df = load_dataset_nations()

# === Train and evaluate ===
model, encoder, scaler, X, y = train_and_evaluate_model(df)

# === Save models ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# === Generate plots ===
generate_visualizations(model, encoder, scaler, X, y)

print("\n📈 Model training, evaluation, and visualizations completed successfully!")
