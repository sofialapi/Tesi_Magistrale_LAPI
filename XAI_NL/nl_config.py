"""Configurazione del modulo XAI_NL.

I percorsi sono relativi alla root del progetto (Tesi_Magistrale_LAPI) e si
possono sovrascrivere da riga di comando. Verificali sulla macchina Tesla.
"""

# --- LLM ---------------------------------------------------------------------
LLM_MODEL = "openai/gpt-oss-20b"
PROMPT_VERSION = "v1"  # cambiala quando modifichi il prompt: invalida la cache

# --- Input dei modelli (identico al training) ---------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Dataset -------------------------------------------------------------------
# processed_dir: immagini con DullRazor + filtro gaussiano (input dei modelli)
# raw_dir:       immagini originali (solo per il pannello mostrato al clinico)
DATASETS = {
    "bcn": {
        "name": "BCN20000",
        "csv": "data/bcn20000/train.csv",
        "processed_dir": "data/processed/BCN",
        "raw_dir": "data/raw/BCN",
        "cnn_ckpt": "outputs/checkpoints/best_cnn_bcn20000.pth",
        "hybrid_ckpt": "outputs/checkpoints/best_hybrid_bcn20000.pth",
        "default_fold": None,
    },
    "ham": {
        "name": "HAM10000",
        "csv": "data/ham10000/ham10000_prepared.csv",
        "processed_dir": "data/processed/HAM",
        "raw_dir": "data/raw/HAM",
        "cnn_ckpt": "outputs/checkpoints/best_cnn_ham10000.pth",
        "hybrid_ckpt": "outputs/checkpoints/best_hybrid_ham10000.pth",
        "default_fold": 1,
    },
}

# --- Output --------------------------------------------------------------------
FIGURE_DIR = "figure/xai_nl"    # PNG con mappe + spiegazione
OUTPUT_DIR = "outputs/xai_nl"   # JSON per caso, riepilogo CSV, cache LLM

# --- Soglie dei descrittori (euristiche, riportate nella tesi) ----------------
CAM_HOT_THRESHOLD = 0.5       # regione "calda": CAM normalizzata >= 0.5
TOP_FRACTION = 0.20           # top-20% dei pixel per il confronto tra mappe
MIN_LESION_ENERGY = 0.50      # sotto: avviso "attivazione fuori dalla lesione"
NEAR_THRESHOLD_MARGIN = 0.10  # |p - tau| sotto: predizione vicina alla soglia

# --- Limiti di lunghezza del testo (parole) ------------------------------------
WORD_LIMITS = {
    "sintesi": 45,
    "resnet50": 60,
    "mobilevit_s": 60,
    "confronto": 40,
    "avvertenza": 30,
}

MODEL_KEYS = ("resnet50", "mobilevit_s")
MODEL_NAMES = {"resnet50": "ResNet-50", "mobilevit_s": "MobileViT-S"}
