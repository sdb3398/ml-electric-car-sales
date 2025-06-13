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

# Analisi variabili categoriche
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\nValori unici per {col}: {df[col].nunique()}")
    print(df[col].value_counts().head())
    
# Analisi delle vendite EV per regione
if set(['region', 'parameter', 'value']).issubset(df.columns):
    vendite_regioni = df[df['parameter'] == 'EV sales'].groupby('region')['value'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='value', y='region', data=vendite_regioni.sort_values('value', ascending=False))
    plt.title('Vendite EV per Regione')
    plt.xlabel('Numero di veicoli venduti')
    plt.ylabel('Regione')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

# Analisi temporale delle vendite EV: Quanti veicoli elettrici sono stati venduti ogni anno a livello globale?
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
    
# Analisi della quota di mercato degli EV per tipo di veicolo
if set(['powertrain', 'year', 'value']).issubset(df.columns):
    quota_mercato = df[df['parameter'] == 'EV stock share'].groupby(['powertrain', 'year'])['value'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=quota_mercato, x='year', y='value', hue='powertrain', marker='o')
    plt.title('Quota di Mercato degli EV per Tipo di Veicolo')
    plt.xlabel('Anno')
    plt.ylabel('Quota di mercato (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Analisi della distribuzione delle vendite EV per tipo di veicolo
if set(['powertrain', 'value']).issubset(df.columns):
    vendite_tipo = df[df['parameter'] == 'EV sales'].groupby('powertrain')['value'].sum().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='powertrain', y='value', data=vendite_tipo)
    plt.title('Vendite EV per Tipo di Veicolo')
    plt.xlabel('Tipo di Veicolo')
    plt.ylabel('Numero di veicoli venduti')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
# Analisi della distribuzione delle vendite EV per regione e tipo di veicolo
if set(['region', 'powertrain', 'value']).issubset(df.columns):
    vendite_region_tipo = df[df['parameter'] == 'EV sales'].groupby(['region', 'powertrain'])['value'].sum().reset_index()
    plt.figure(figsize=(12, 8))
    sns.barplot(x='value', y='region', hue='powertrain', data=vendite_region_tipo)
    plt.title('Vendite EV per Regione e Tipo di Veicolo')
    plt.xlabel('Numero di veicoli venduti')
    plt.ylabel('Regione')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()
    
# Analisi della distribuzione delle vendite EV per anno e tipo di veicolo
if set(['year', 'powertrain', 'value']).issubset(df.columns):
    vendite_anno_tipo = df[df['parameter'] == 'EV sales'].groupby(['year', 'powertrain'])['value'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=vendite_anno_tipo, x='year', y='value', hue='powertrain', marker='o')
    plt.title('Vendite EV per Anno e Tipo di Veicolo')
    plt.xlabel('Anno')
    plt.ylabel('Numero di veicoli venduti')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    



