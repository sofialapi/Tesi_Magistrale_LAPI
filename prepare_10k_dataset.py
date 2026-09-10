import os
import h5py
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.preprocessing.image_preprocessing import dull_razor, standardize_image

# Modifica questi due path con quelli trovati con il comando find al Passo 1
ORIGINAL_METADATA_PATH = "/mnt/sdb1/workspace/sofialapi/data/raw/train-metadata.csv"
ORIGINAL_H5_PATH = "/mnt/sdb1/workspace/sofialapi/data/raw/train-image.hdf5"

OUTPUT_METADATA_CSV = "/mnt/sdb1/workspace/sofialapi/Tesi_Magistrale_LAPI/data/raw/metadata_isic_10k.csv"
OUTPUT_PROCESSED_DIR = "/mnt/sdb1/workspace/sofialapi/Tesi_Magistrale_LAPI/data/processed_10k"

os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)

print("Caricamento metadati completi...")
df_all = pd.read_csv(ORIGINAL_METADATA_PATH)

# Estrazione: TUTTI i positivi + negativi campionati fino a 10.000
df_pos = df_all[df_all['target'] == 1]
n_pos = len(df_pos)
n_neg = 10000 - n_pos

df_neg = df_all[df_all['target'] == 0].sample(n=n_neg, random_state=42)
df_10k = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)

df_10k.to_csv(OUTPUT_METADATA_CSV, index=False)
print(f"Dataset 10k campionato: {n_pos} positivi (melanomi) e {n_neg} negativi (benigni).")

# Estrazione ed elaborazione da file HDF5
print(f"Estrazione e preprocessing di 10.000 immagini da {ORIGINAL_H5_PATH}...")
with h5py.File(ORIGINAL_H5_PATH, 'r') as h5_file:
    for isic_id in tqdm(df_10k['isic_id']):
        out_path = os.path.join(OUTPUT_PROCESSED_DIR, f"{isic_id}.jpg")
        if os.path.exists(out_path):
            continue
            
        # Decodifica immagine binaria JPG contenuta nel nodo HDF5
        img_bytes = h5_file[isic_id][()]
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Pipeline: Dull Razor + Standardize
        # Per evitare I/O su disco, applichiamo Dull Razor in memoria
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        clean = cv2.inpaint(img_np, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        
        final_img = standardize_image(clean, target_size=(224, 224))
        cv2.imwrite(out_path, final_img)

print("Elaborazione 10k completata con successo.")