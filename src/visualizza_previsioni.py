import pandas as pd
import matplotlib.pyplot as plt

# === Carica il file locale
csv_path = "data/EV_sales_predictions_2024.csv"
df = pd.read_csv(csv_path)

# === Tabella ordinata per vendite EV previste
print("\n📊 Prime 10 righe del file:\n")
print(df.sort_values("predicted_sales", ascending=False).head(10))

# === Grafico Top 15 nazioni (aggregato su powertrain)
top15 = df.groupby("nation")["predicted_sales"].sum().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 6))
top15.plot(kind="bar")
plt.title("Top 15 Nazioni per Vendite EV Previste nel 2024")
plt.ylabel("Numero di Veicoli")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/Top_15_Countries_by_total_EV_Sales.png")
plt.close()

