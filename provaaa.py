from dataset import load_dataset_nations

df = load_dataset_nations()
df = df[df["parameter"] == "EV sales"]

# Mostra gli anni unici presenti
print("📅 Anni disponibili nel dataset:")
print(sorted(df["year"].unique(), reverse=True))
