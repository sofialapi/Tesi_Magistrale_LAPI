#gestione del caricamento dei file CSV, pulizia dei dati strutturati tabulari di ISIC2024, fusione finale
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class ClinicalMetadataProcessor:
    def __init__(self):
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Feature estratte dalla letteratura (es. Paper Multimodale ISIC 2024)
        self.num_features = ['age_approx', 'tbp_lv_symm_2axis', 'tbp_lv_eccentricity', 'tbp_lv_areaMM2']
        self.cat_features = ['sex', 'anatom_site_general']

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Pulisce, imputa i valori mancanti e standardizza le feature cliniche.
        """
        # Estrai le colonne di interesse
        df_num = df[self.num_features].copy()
        df_cat = df[self.cat_features].copy()
        
        # Imputazione e Scaling dei dati numerici
        num_imputed = self.imputer_num.fit_transform(df_num)
        num_scaled = self.scaler.fit_transform(num_imputed)
        
        # Imputazione e One-Hot Encoding dei dati categorici
        cat_imputed = self.imputer_cat.fit_transform(df_cat)
        cat_encoded = self.encoder.fit_transform(cat_imputed)
        
        # Concatenazione dei vettori di feature cliniche
        clinical_features = np.hstack((num_scaled, cat_encoded))
        return clinical_features