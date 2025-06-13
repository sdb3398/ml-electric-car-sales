"""
Colonne del dataset:
- region: Area geografica di raccolta dati (es. Australia)
- category: Natura del dato (es. Historical, Projection)
- parameter: Tipo di metrica (es. EV sales, EV stock share)
- mode: Modalità di trasporto (es. Cars)
- powertrain: Tipo di EV (BEV, PHEV, EV)
- year: Anno del dato (es. 2011, 2012)
- unit: Unità di misura (es. Vehicles, percent)
- value: Valore registrato
"""

from dataset import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il DataFrame
df = load_dataset()

# Statistiche descrittive generali
print("\nStatistiche descrittive:")
print(df.describe(include='all'))

# Analisi valori nulli (heatmap)
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Mappa valori nulli')
plt.show()

# Analisi variabili numeriche
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribuzione di {col}')
    plt.show()

# Analisi variabili categoriche
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\nValori unici per {col}: {df[col].nunique()}")
    print(df[col].value_counts().head())

# Correlazione tra variabili numeriche
if len(num_cols) > 1:
    plt.figure(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matrice di correlazione')
    plt.show()

# Analisi temporale delle vendite EV (se presenti le colonne 'parameter', 'year', 'value')
if set(['parameter', 'year', 'value']).issubset(df.columns):
    vendite = df[df['parameter'] == 'EV sales']
    vendite_anno = vendite.groupby('year')['value'].sum().reset_index()
    plt.figure()
    plt.plot(vendite_anno['year'], vendite_anno['value'], marker='o')
    plt.title('EV Sales Globali per Anno')
    plt.xlabel('Anno')
    plt.ylabel('Numero di veicoli venduti')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

