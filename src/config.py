import os

# Percorso radice del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path generici per la pipeline dei dati
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# Sottocartelle specifiche per Dataset (Novità)
RAW_ISIC_DIR = os.path.join(RAW_DATA_DIR, "ISIC")
RAW_HAM_DIR = os.path.join(RAW_DATA_DIR, "HAM")
RAW_BCN_DIR = os.path.join(RAW_DATA_DIR, "BCN")

PROCESSED_ISIC_DIR = os.path.join(PROCESSED_DATA_DIR, "ISIC")
PROCESSED_HAM_DIR = os.path.join(PROCESSED_DATA_DIR, "HAM")
PROCESSED_BCN_DIR = os.path.join(PROCESSED_DATA_DIR, "BCN")

# Parametri di Preprocessing delle immagini
TARGET_IMAGE_SIZE = (224, 224)