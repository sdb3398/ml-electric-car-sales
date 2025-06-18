
# 🚗 Electric Vehicle Sales Prediction (2010–2024)

Questo progetto ha l'obiettivo di costruire un modello di regressione per prevedere le vendite di veicoli elettrici (EV) a partire da dati storici nazionali, suddivisi per anno, tipo di motorizzazione e altri fattori.

---

## 📦 Dataset

- Fonte: Kaggle – *Electric car sales (2010–2024)*
- Dati pre-trattati: filtrati per `EV sales`, solo nazioni, senza aggregati come `World` o `Europe`
- Colonne rilevanti:
  - `nation`, `year`, `powertrain`
  - `value`: numero di EV venduti

---

## 🧼 Preprocessing

1. **Filtro sul parametro `EV sales`**.
2. **Creazione variabili temporali**:
   - `sales_last_year`: vendite dell’anno precedente (lag)
   - `growth_rate`: tasso di crescita percentuale anno su anno
3. **Winsorization del target**:
   - Per limitare outlier estremi (1° e 99° percentile)
4. **Trasformazione logaritmica del target** con `log1p`, e ritorno con `expm1`
5. **Scaling robusto** delle feature numeriche (`year`, `sales_last_year`, `growth_rate`)
6. **One-hot encoding** di `nation` e `powertrain`

---

## 🧠 Modello

- Modello principale: `XGBRegressor` incapsulato in `TransformedTargetRegressor` per gestire il log-transform
- Iperparametri default:
  - `n_estimators=100`, `random_state=42`
- Feature usate:
  - `year_scaled`, `sales_last_year_scaled`, `growth_scaled`
  - Dummies `nation_*`, `powertrain_*`

---

## 🔍 Valutazione

### ⚙️ Cross-validation 5-fold
Metriche calcolate:
- **MAE**: Errore medio assoluto
- **RMSE**: Radice dell'errore quadratico medio
- **R²**: Varianza spiegata
- **MAPE**: Errore percentuale medio
- **MedAE**: Errore mediano
- **MaxError**: Massimo errore assoluto

### 📈 Risultati finali (modello esteso)

| Metrica     | Valore       |
|-------------|--------------|
| MAE         | ~2.528       |
| RMSE        | ~11.494      |
| R²          | 0.9700       |
| MAPE        | 0.15         |
| MedAE       | ~87.74       |
| MaxError    | ~116.214     |

---

## ❓ Come sappiamo che NON stiamo overfittando?

- Utilizziamo **cross-validation** su 5 fold → il modello è testato su porzioni di dati mai viste
- L’**R² test è allineato** tra training e validation fold → no gap sospetto
- Le metriche come **MAE, MedAE e MAPE sono molto basse** → il modello è preciso anche nei casi medi e non solo sui grandi volumi
- **Feature importance ben distribuite** → il modello non si appoggia solo a una variabile "magica"
- **Winsorization e scaling** rendono il modello meno sensibile agli outlier

---

## ✅ Conclusioni

Il modello è:

- **Stabile**
- **Interpretabile**
- **Accurato (R² > 0.97)**
- **Pronto per essere salvato e integrato**

---


