import pandas as pd

# Contoh data
data = {'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA', 'ATA']}
df = pd.DataFrame(data)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['ChestPainType'])


df_encoded = pd.get_dummies(df, columns=['ChestPainType']).astype(int)

print(df_encoded)
