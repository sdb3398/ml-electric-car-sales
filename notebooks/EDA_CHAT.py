# EDA PER CLASSIFICATION AND REGRESSION
import pandas as pd
import matplotlib.pyplot as plt
from dataset import load_dataset_nations

df = load_dataset_nations()

# Selezione metrica EV sales
ev_sales = df[df["parameter"] == "EV sales"]

# Aggrega vendite per anno
sales_per_year = ev_sales.groupby("year")["value"].sum()

# Grafico
plt.figure(figsize=(10, 5))
sales_per_year.plot(marker="o")
plt.title("Global EV Sales per Year")
plt.xlabel("Year")
plt.ylabel("Vehicles sold")
plt.grid(True)
plt.show()

top_countries = ev_sales.groupby("nation")["value"].sum().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 6))
top_countries.plot(kind="bar")
plt.title("Top 15 countries by total EV sales")
plt.ylabel("Total EV Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

powertrain_sales = ev_sales.groupby(["year", "powertrain"])["value"].sum().unstack()

powertrain_sales.plot(figsize=(10, 5), marker="o")
plt.title("EV Sales by Powertrain Over Time")
plt.ylabel("Vehicles")
plt.grid(True)
plt.show()

