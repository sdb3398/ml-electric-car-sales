import pandas as pd
import os
import kagglehub

def load_dataset_full():
    # Scarica il dataset (se non già scaricato)
    path = kagglehub.dataset_download("jainaru/electric-car-sales-2010-2024")
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    csv_path = os.path.join(path, files[0])
    df = pd.read_csv(csv_path)
    return df

def load_dataset_nations():
    df = load_dataset_full()
    continenti_veri = ['Europe', 'Asia', 'Africa', 'Oceania', 'Americas']
    aggregati = ['World', 'Rest of the world', 'EU27']
    escludi = continenti_veri + aggregati
    df_nazioni = df[~df['region'].isin(escludi)].copy()
    df_nazioni.rename(columns={'region': 'nation'}, inplace=True)
    return df_nazioni

if __name__ == "__main__":
    df_full = load_dataset_full()
    print("--- Dataset completo ---")
    print(df_full.head())
    print("\n🧱 Shape:", df_full.shape)
    print("\n📝 Colonne:", df_full.columns)
    print("\n🔍 Tipi di dato:")
    print(df_full.dtypes)
    print("\n❓ Valori nulli per colonna:")
    print(df_full.isnull().sum())
    print("\nValori unici in 'region':")
    print(sorted(df_full['region'].unique()))
    print("\n--- Solo nazioni ---")
    df_nazioni = load_dataset_nations()
    print(df_nazioni.head())
    print("\n🧱 Shape:", df_nazioni.shape)
    print("\n📝 Colonne:", df_nazioni.columns)
    print("\n🔍 Tipi di dato:")
    print(df_nazioni.dtypes)
    print("\n❓ Valori nulli per colonna:")
    print(df_nazioni.isnull().sum())
    print("\nValori unici in 'nation':")
    print(sorted(df_nazioni['nation'].unique()))
