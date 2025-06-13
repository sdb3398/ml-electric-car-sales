#prove

from dataset import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

df = load_dataset()

# Trova tutti i valori unici della colonna 'region' per identificare continenti e paesi
if 'region' in df.columns:
    print("\nValori unici in 'region':")
    print(sorted(df['region'].unique()))