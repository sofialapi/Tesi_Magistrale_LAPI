import os

# Percorso radice del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path per la pipeline dei dati (Configurazione Locale per i Test)
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# Parametri di Preprocessing delle immagini
TARGET_IMAGE_SIZE = (224, 224)

# Nota: Quando ti sposti sul server con i dati reali, cambia il path così:
# RAW_DATA_DIR = "/datasets/skin_cancer/ISIC2024/raw"