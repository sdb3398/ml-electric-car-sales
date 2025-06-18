from dataset import load_dataset_nations
from src.model_training import train_and_evaluate_model
import joblib
import os

# === Load dataset ===
df = load_dataset_nations()

# === Train model and evaluate ===
model, encoder, scaler = train_and_evaluate_model(df)

# === Save artifacts ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\n\U0001F4C8 Model training and evaluation completed successfully!")