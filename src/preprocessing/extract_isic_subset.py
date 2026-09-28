import os
import h5py
import io
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

ISIC_RAW_DIR = "/mnt/sdb1/workspace/sofialapi/Tesi_Magistrale_LAPI/data/raw/ISIC"
H5_PATH = os.path.join(ISIC_RAW_DIR, "train-image.hdf5")
CSV_PATH = os.path.join(ISIC_RAW_DIR, "train-metadata.csv")
OUTPUT_CSV_PATH = os.path.join(ISIC_RAW_DIR, "metadata_isic_subset.csv")

def extract_subset(benign_sample_size: int = 10000, random_state: int = 42):
    print("Lettura metadati ISIC 2024...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Filtro: tutti i melanomi (target=1) + subset bilanciato di benigni (target=0)
    df_malignant = df[df['target'] == 1].copy()
    df_benign = df[df['target'] == 0].sample(n=benign_sample_size, random_state=random_state).copy()
    
    df_subset = pd.concat([df_malignant, df_benign], axis=0).reset_index(drop=True)
    print(f"Campioni selezionati: Totale={len(df_subset)} | Maligni={len(df_malignant)} | Benigni={len(df_benign)}")
    
    # Salva il nuovo CSV di lavoro
    df_subset.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Metadati del subset salvati in: {OUTPUT_CSV_PATH}")
    
    # 2. Estrazione immagini dall'archivio HDF5
    print("Apertura archivio HDF5 ed estrazione immagini...")
    selected_ids = set(df_subset['isic_id'].values)
    
    with h5py.File(H5_PATH, 'r') as h5:
        for isic_id in tqdm(selected_ids, desc="Estrazione JPEG"):
            out_img_path = os.path.join(ISIC_RAW_DIR, f"{isic_id}.jpg")
            
            # Salta se già presente
            if os.path.exists(out_img_path):
                continue
                
            if isic_id in h5:
                img_bytes = io.BytesIO(h5[isic_id][()])
                img = Image.open(img_bytes)
                img.save(out_img_path, format="JPEG")
            else:
                print(f"[ATTENZIONE] ID non trovato in HDF5: {isic_id}")
                
    print(f"Estrazione completata con successo in {ISIC_RAW_DIR}.")

if __name__ == "__main__":
    extract_subset(benign_sample_size=10000)
