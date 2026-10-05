"""Configurazione dell'analisi SHAP sui modelli multimodali ISIC 2024 (Casi 3 e 4)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(ROOT, "data", "raw", "ISIC", "metadata_isic_subset.csv")
IMAGE_DIR = os.path.join(ROOT, "data", "processed", "ISIC")
CKPT_DIR = os.path.join(ROOT, "outputs", "checkpoints")

OUT_DIR = os.path.join(ROOT, "outputs", "shap")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
FIG_DIR = os.path.join(ROOT, "figure", "isic2024", "shap")

# Stessi parametri di src/training/train.py: lo split deve essere identico.
N_FOLDS = 5
SPLIT_SEED = 42

# Casi analizzati (solo i multimodali hanno un ramo tabulare).
CASES = {
    "cnn_multimodal": {"backbone": "cnn", "label": "Caso 3 - ResNet-50 + tabulare", "short": "ResNet-50 + tab."},
    "hybrid_multimodal": {"backbone": "hybrid", "label": "Caso 4 - MobileViT-S + tabulare", "short": "MobileViT-S + tab."},
}

# Background per SHAP interventional: campione casuale del fold di training.
BACKGROUND_SIZE = 200
BACKGROUND_SEED = 1234

# Batch per estrazione embedding (immagini) e per il calcolo delle coalizioni.
IMG_BATCH = 64
SAMPLE_CHUNK = 16
NUM_WORKERS = 4

# Variabili originali (giocatori SHAP): le colonne one-hot di una stessa
# variabile categoriale formano un unico giocatore.
NUM_VARS = ["age_approx", "tbp_lv_symm_2axis", "tbp_lv_eccentricity", "tbp_lv_areaMM2"]
CAT_VARS = ["sex", "anatom_site_general"]
VAR_ORDER = NUM_VARS + CAT_VARS

VAR_LABELS = {
    "age_approx": "Età",
    "tbp_lv_symm_2axis": "Asimmetria (2 assi)",
    "tbp_lv_eccentricity": "Eccentricità",
    "tbp_lv_areaMM2": "Area (mm²)",
    "sex": "Sesso",
    "anatom_site_general": "Sede anatomica",
}

CAT_LABELS = {
    "male": "M", "female": "F",
    "posterior torso": "tronco post.", "anterior torso": "tronco ant.",
    "lower extremity": "arto inf.", "upper extremity": "arto sup.",
    "head/neck": "testa/collo",
}

# Colori coerenti con le figure dei Capitoli 4-5 (blu ResNet-50, arancio MobileViT-S).
CASE_COLORS = {"cnn_multimodal": "#1f77b4", "hybrid_multimodal": "#ff7f0e"}
CLASS_COLORS = {0: "#9a9a9a", 1: "#c0392b"}
