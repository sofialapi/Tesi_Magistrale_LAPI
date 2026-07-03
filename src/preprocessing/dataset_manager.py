#gestione del caricamento dei file CSV, pulizia dei dati strutturati tabulari di ISIC2024, fusione finale
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.config import RAW_DATA_DIR

class ClinicalMetadataProcessor:
    def __init__(self):
        # Definiamo gli imputer per gestire i dati mancanti (Missing Values)
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        
        # Standardizzatore per scalare i valori numerici (Media=0, Varianza=1)
        self.scaler = StandardScaler()
        
        # Codificatore One-Hot per trasformare le stringhe categoriali in vettori binari
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Feature geometriche e demografiche estratte dalla letteratura (ISIC 2024 Challenge)
        self.num_features = ['age_approx', 'tbp_lv_symm_2axis', 'tbp_lv_eccentricity', 'tbp_lv_areaMM2']
        self.cat_features = ['sex', 'anatom_site_general']
        
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Esegue il fit e la trasformazione combinata delle feature numeriche e categoriali.
        """
        # 1. Isolamento e pulizia delle colonne numeriche
        df_num = df[self.num_features].copy()
        num_imputed = self.imputer_num.fit_transform(df_num)
        num_scaled = self.scaler.fit_transform(num_imputed)
        
        # 2. Isolamento e pulizia delle colonne categoriali
        df_cat = df[self.cat_features].copy()
        cat_imputed = self.imputer_cat.fit_transform(df_cat)
        cat_encoded = self.encoder.fit_transform(cat_imputed)
        
        # 3. Concatenazione orizzontale dei due vettori (Feature Fusion di base)
        processed_features = np.hstack((num_scaled, cat_encoded))
        return processed_features

def load_and_process_metadata(csv_path: str) -> tuple:
    """
    Funzione di orchestrazione per caricare il CSV ed estrarre feature e label.
    Accetta il percorso assoluto del file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Impossibile trovare il file dei metadati in: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Metadati caricati correttamente. Righe: {df.shape[0]}, Colonne: {df.shape[1]}")
    
    processor = ClinicalMetadataProcessor()
    X_clinical = processor.fit_transform(df)
    
    y = df['target'].values if 'target' in df.columns else None
    
    return X_clinical, y, df['isic_id'].values