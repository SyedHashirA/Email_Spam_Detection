import pandas as pd

df = pd.read_csv("/backend/data/newspam.csv")

# Map 1 → spam, 0 → ham
df["label"] = df["label"].map({1: "spam", 0: "ham"})

df.to_csv("/backend/data/newspam.csv", index=False)
print("✅ Converted! First few rows:")
print(df.head())
