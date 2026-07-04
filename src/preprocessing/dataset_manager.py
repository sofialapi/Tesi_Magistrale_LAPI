import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.preprocessing.augmentation import DermalImageAugmentor

class ClinicalMetadataProcessor:
    """Gestore del preprocessing dei dati clinici tabulari (Imputazione, Z-score, One-Hot)."""
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        self.num_cols = ['age_approx', 'tbp_lv_symm_2axis', 'tbp_lv_eccentricity', 'tbp_lv_areaMM2']
        self.cat_cols = ['sex', 'anatom_site_general']
        
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df_num = pd.DataFrame(self.num_imputer.fit_transform(df[self.num_cols]), columns=self.num_cols)
        df_cat = pd.DataFrame(self.cat_imputer.fit_transform(df[self.cat_cols]), columns=self.cat_cols)
        
        X_num_scaled = self.scaler.fit_transform(df_num)
        X_cat_encoded = self.encoder.fit_transform(df_cat)
        
        return np.hstack((X_num_scaled, X_cat_encoded))


class DermalMultimodalDataset(Dataset):
    """
    Dataset PyTorch flessibile per la classificazione delle lesioni cutanee.
    Supporta due modalità:
    - 'multimodal': restituisce (immagine, metadati_clinici, label)
    - 'image_only': restituisce (immagine, label) per dataset privi di metadati (BCN, HAM)
    """
    def __init__(self, image_ids, labels, image_dir, clinical_matrix=None, target_size=(224, 224), mode='multimodal', is_training=True):
        self.image_ids = image_ids
        self.labels = labels
        self.image_dir = image_dir
        self.clinical_matrix = clinical_matrix
        self.mode = mode.lower()
        self.is_training = is_training
        
        # Validazione di sicurezza della modalità prescelta
        if self.mode not in ['multimodal', 'image_only']:
            raise ValueError("La modalità del dataset deve essere impostata su 'multimodal' o 'image_only'.")
            
        if self.mode == 'multimodal' and self.clinical_matrix is None:
            raise ValueError("La modalità 'multimodal' richiede obbligatoriamente il passaggio della matrice dei metadati clinici.")
            
        # Inizializzazione dell'Augmentor on-the-fly configurato in precedenza
        self.augmentor = DermalImageAugmentor(target_size=target_size, is_training=self.is_training)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Recupero dell'ID e caricamento dell'immagine fisica dal disco (data/processed/)
        img_id = self.image_ids[idx]
        
        # Gestione flessibile dell'estensione del file
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERRORE DATASET] Immagine non trovata nel percorso: {img_path}")
            
        # Lettura in formato OpenCV e conversione cromatica corretta
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Applicazione dell'Augmentor (conversione in tensore, scaling, resize, aumenti geometrici/fotometrici)
        img_tensor = self.augmentor(img_rgb)
        
        # 3. Recupero della label del target (0 o 1) convertito in tensore PyTorch
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 4. Restituzione condizionale in base alla modalità selezionata
        if self.mode == 'multimodal':
            # Estrazione della riga corrispondente e conversione in tensore float32
            clinical_vector = self.clinical_matrix[idx]
            clinical_tensor = torch.tensor(clinical_vector, dtype=torch.float32)
            return img_tensor, clinical_tensor, label_tensor
        else:
            # Modalità Image-Only: esclude completamente la parte tabulare
            return img_tensor, label_tensor


def load_and_process_metadata(csv_path: str):
    """Funzione helper per caricare e preparare i metadati clinici di ISIC."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV non trovato: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Metadati caricati correttamente. Righe: {df.shape[0]}, Colonne: {df.shape[1]}")
    
    processor = ClinicalMetadataProcessor()
    X_clinical = processor.fit_transform(df)
    
    y = df['target'].values
    image_ids = df['isic_id'].values
    
    return X_clinical, y, image_ids

def create_multimodal_dataloader(dataset: Dataset, batch_size: int = 4, shuffle: bool = True, num_workers: int = 2) -> DataLoader:
    """
    Fabbrica e configura un'istanza di DataLoader PyTorch ottimizzata per l'ambiente locale.
    
    Parametri:
    - dataset: Istanza di DermalMultimodalDataset (in modalità multimodal o image_only).
    - batch_size: Numero di campioni per batch (impostato basso di default per la VM locale).
    - shuffle: Se True, mescola i dati ad ogni epoca (fondamentale in training).
    - num_workers: Numero di core CPU dedicati al caricamento parallelo in background.
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False # Ottimizza il passaggio verso la GPU
    )
    return dataloader