import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

data_dir = "data/ham10000"
meta_path = os.path.join(data_dir, "HAM10000_metadata.csv")
img_dir = os.path.join(data_dir, "processed_images")

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"File non trovato: {meta_path}")

df = pd.read_csv(meta_path)
df["target"] = (df["dx"] == "mel").astype(int)

df["filepath"] = df["image_id"].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
df["exists"] = df["filepath"].apply(os.path.exists)
df = df[df["exists"]].reset_index(drop=True)

print(f"Campioni validi preelaborati: {len(df)}")
print(f"Distribuzione: Negativi={sum(df['target'] == 0)}, Melanomi={sum(df['target'] == 1)}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df["target"], groups=df["lesion_id"])):
    df.loc[val_idx, "fold"] = fold + 1

out_csv = os.path.join(data_dir, "ham10000_prepared.csv")
df.to_csv(out_csv, index=False)
print(f"Salvato con successo in: {out_csv}")
