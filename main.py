from src.dataset import load_dataset_nations
from src.regression_upgraded import train_and_evaluate_model
from src.predict_ev_sales import predict_sales
from src.batch_predict_ev_sales import run_batch_prediction

if __name__ == "__main__":
    print("🔍 Caricamento dataset...")
    df = load_dataset_nations()

    print("🚀 Addestramento modello e valutazione...")
    model, encoder, scaler = train_and_evaluate_model(df)

    print("🔮 Predizione su un singolo esempio (Italia 2024, BEV)...")
    example = {"nation": "Italy", "powertrain": "BEV", "year": 2024, "sales_last_year": 165000, "growth_rate": 0.15}
    pred = predict_sales(example, model, encoder, scaler)
    print(f"✅ Predicted EV Sales: {int(pred):,} vehicles")

    print("📊 Esecuzione predizione batch...")
    run_batch_prediction()

    print("✅ Tutto completato!")
