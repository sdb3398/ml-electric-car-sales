import pandas as pd
import os
import kagglehub

# Scarica il dataset (se non già scaricato)
path = kagglehub.dataset_download("jainaru/electric-car-sales-2010-2024")

# Trova il file CSV (assumendo che ci sia un solo CSV)
files = [f for f in os.listdir(path) if f.endswith(".csv")]
csv_path = os.path.join(path, files[0])

# Carica il dataset
df = pd.read_csv(csv_path)

# Mostra le prime righe
print(df.head())
print("\n🧱 Shape:", df.shape)
print("\n📝 Colonne:", df.columns)
print("\n🔍 Tipi di dato:")
print(df.dtypes)

print("\n❓ Valori nulli per colonna:")
print(df.isnull().sum())
