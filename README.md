# 🔋 Electric Vehicle Sales Forecasting

Questo progetto mira a prevedere le vendite di veicoli elettrici (EV) per nazione e tipo di motore (BEV, PHEV) a partire dai dati storici 2010–2023.

---

## 📁 Struttura del progetto

```
ml-electric-car-sales/
├── models/                # Modelli salvati (.gitkeep presente)
│
├── notebooks/             # Notebook e script EDA
│   ├── EDA.py
│   └── EDA_2.py
│
├── src/                   # Codice sorgente
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── regression_models.py
│   ├── regression_outliers.py
│   ├── regression_advanced.py
│   ├── regression_upgraded.py
│   ├── predict_ev_sales.py
│   ├── batch_predict_ev_sales.py
│   └── visualizza_previsioni.py
│
├── plots/                 # Grafici
│
├── dataset.py             # load e manipolazione dataset
├── main.py                # Script principale di orchestrazione
├── requirements.txt       # Dipendenze
└── .gitignore
```

---

## 🚀 Pipeline

1. **Analisi Esplorativa** (`EDA.py`)
2. **Preparazione dati** (`dataset.py`)
3. **Modellazione** (`regression_advanced.py` o `regression_upgraded.py`)
4. **Trattamento outlier** (`regression_outliers.py`)
5. **Salvataggio del modello**
6. **Predizione singola** (`predict_ev_sales.py`)
7. **Batch prediction** (`batch_predict_ev_sales.py`)
8. **Visualizzazione risultati** (`visualizza_previsioni.py`)

---

## 📦 Dipendenze principali

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost

Installa tutto con:

```bash
pip install -r requirements.txt
```
