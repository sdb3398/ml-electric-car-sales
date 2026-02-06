# 📈 Electric Vehicle Sales Forecasting (2010–2024)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Previsione delle vendite di veicoli elettrici a livello globale attraverso modelli di regressione avanzata su dati storici reali.

---

## 🧠 Obiettivo

Utilizzare dati storici sulle vendite di veicoli elettrici per:

- Effettuare un'analisi esplorativa (EDA),
- Addestrare un modello predittivo (XGBoost con regressione trasformata),
- Valutare le performance con metodi robusti,
- Prevedere le vendite del 2024 per ciascun Paese e powertrain,
- Visualizzare le previsioni e le metriche in modo chiaro e interpretabile.

---

## 📁 Struttura del progetto

```
ml-electric-car-sales/
├── models/                # Modelli salvati (.gitkeep presente)
│
├── notebooks/             # script EDA
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
│   ├── visualizations.py
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

## 🔍 Dataset

Dati provenienti da [Kaggle - EV Sales Dataset](https://www.kaggle.com/datasets/jainaru/electric-car-sales-2010-2024), contenenti vendite dal 2010 al 2023 per nazione e tipo di powertrain.

---

## ⚙️ Funzionalità principali

- Feature Engineering (Lag e Growth Rate)
- Winsorization su outlier target
- Preprocessing: encoding e scaling
- Regressione XGBoost con trasformazione logaritmica
- K-Fold Cross Validation
- Salvataggio modelli e previsioni
- Predizione vendite 2024
- Visualizzazione performance e importanza delle feature

---

## 📊 Metriche

**Cross Validation:**

- MAE: ~2,529  
- RMSE: ~11,494  
- R²: ~0.97  
- MAPE: ~15%

**Test Set:**

- MAE: ~3,514  
- RMSE: ~13,215  
- R²: ~0.97  
- MAPE: ~17%

Le metriche su training e test sono molto simili → **il modello generalizza bene** e non mostra segnali evidenti di overfitting.

---

## 🚀 Avvio rapido

### 1. Clona il progetto
```bash
git clone https://github.com/SimoneDalBen/ml-electric-car-sales.git
cd ml-electric-car-sales
```

### 2. Crea ambiente e installa dipendenze
```bash
python -m venv venv
venv\Scripts\activate  # su Windows
pip install -r requirements.txt
```

### 3. Esgui training + valutazione
```bash
python main.py
```

### 4. Esegui le previsioni per il 2024
```bash
python src/batch_predict_ev_sales.py
```

## 📦 Dipendenze principali

- Python 3.11+
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- joblib

Installa tutto con:

```bash
pip install -r requirements.txt
```

---

## 👤 Autore

**Simone Dal Ben**  
📧 [simonedalben38@gmail.com](mailto:simonedalben38@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/simone-dal-ben)  

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file [LICENSE](LICENSE) per maggiori dettagli.
