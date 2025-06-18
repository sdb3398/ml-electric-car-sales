"""
Colonne del dataset:
- nation: Area geografica di raccolta dati (es. Australia)
- category: Natura del dato (es. Historical, Projection)
- parameter: Tipo di metrica (es. EV sales, EV stock share)
- mode: Modalità di trasporto (es. Cars)
- powertrain: Tipo di EV (BEV, PHEV, EV)
- year: Anno del dato (es. 2011, 2012)
- unit: Unità di misura (es. Vehicles, percent)
- value: Valore registrato
"""

from dataset import load_dataset_full, load_dataset_nations
import matplotlib.pyplot as plt
import seaborn as sns

# Carica entrambi i DataFrame
full_df = load_dataset_full()
df = load_dataset_nations()

# Statistiche descrittive generali (solo nazioni)
print("\nStatistiche descrittive (solo nazioni):")
print(df.describe(include='all'))

# Analisi variabili categoriche (solo nazioni)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\nValori unici per {col}: {df[col].nunique()}")
    print(df[col].value_counts().head())
    
# Analisi delle vendite EV per nazione
if set(['nation', 'parameter', 'value']).issubset(df.columns):
    vendite_nationi = df[(df['parameter'] == 'EV sales')]
    somma_nationi = vendite_nationi.groupby('nation')['value'].sum().reset_index()
    totale_nationi = somma_nationi['value'].sum()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='value', y='nation', data=somma_nationi.sort_values('value', ascending=False))
    plt.title('Vendite EV per nazione')
    plt.xlabel('Numero di veicoli venduti')
    plt.ylabel('Nazione')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()

# Analisi delle vendite EV per nazione (solo unità 'Vehicles')
if set(['nation', 'parameter', 'value', 'unit']).issubset(df.columns):
    vendite_nationi = df[(df['parameter'] == 'EV sales') & (df['unit'] == 'Vehicles')]
    somma_nationi = vendite_nationi.groupby('nation')['value'].sum().reset_index()
    somma_nationi = somma_nationi.sort_values('value', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='value', y='nation', data=somma_nationi)
    plt.title('Vendite EV per nazione (solo unità Vehicles, somma 2010-2024)')
    plt.xlabel('Numero di veicoli venduti (2010-2024)')
    plt.ylabel('Nazione')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()
    # Stampa le prime nazioni per controllo
    print(somma_nationi.head(10))

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

# Analisi della distribuzione delle vendite EV per tipo di veicolo (escludendo 'EV' generico)
if set(['powertrain', 'value', 'parameter']).issubset(df.columns):
    tipi_specifici = ['BEV', 'PHEV', 'FCEV']
    vendite_tipo = df[(df['parameter'] == 'EV sales') & (df['powertrain'].isin(tipi_specifici))]
    vendite_tipo = vendite_tipo.groupby('powertrain')['value'].sum().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='powertrain', y='value', data=vendite_tipo)
    plt.title('Vendite EV per Tipo di Veicolo (solo tipi specifici)')
    plt.xlabel('Tipo di Veicolo')
    plt.ylabel('Numero di veicoli venduti')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
# Analisi della distribuzione delle vendite EV per natione e tipo di veicolo
if set(['nation', 'powertrain', 'value']).issubset(df.columns):
    vendite_nation_tipo = df[df['parameter'] == 'EV sales'].groupby(['nation', 'powertrain'])['value'].sum().reset_index()
    plt.figure(figsize=(12, 8))
    sns.barplot(x='value', y='nation', hue='powertrain', data=vendite_nation_tipo)
    plt.title('Vendite EV per natione e Tipo di Veicolo')
    plt.xlabel('Numero di veicoli venduti')
    plt.ylabel('natione')
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

# Analisi temporale globale delle vendite EV (World)
if set(['region', 'parameter', 'value', 'year']).issubset(full_df.columns):
    vendite_world = full_df[(full_df['parameter'] == 'EV sales') & (full_df['region'] == 'World')]
    if not vendite_world.empty:
        vendite_anno = vendite_world.groupby('year')['value'].sum().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(vendite_anno['year'], vendite_anno['value'], marker='o')
        plt.title('Vendite EV Globali per Anno (World)')
        plt.xlabel('Anno')
        plt.ylabel('Numero di veicoli venduti')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



