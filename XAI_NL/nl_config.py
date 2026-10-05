"""Configurazione del modulo XAI_NL (percorsi verificati sulla macchina Tesla).

Percorsi relativi alla root del progetto (~/Tesi_Magistrale_LAPI).

Checkpoint: per spiegare solo immagini di VALIDAZIONE si usano i checkpoint
per fold. Un checkpoint "{fold}" viene completato con il numero del fold;
un dizionario {fold: percorso} elenca i soli fold disponibili.
"""

# --- LLM ---------------------------------------------------------------------
LLM_MODEL = "openai/gpt-oss-20b"
PROMPT_VERSION = "v2"  # cambiala quando modifichi il prompt: invalida la cache

# --- Input dei modelli (identico al val_tf dei training) ----------------------
# PIL .convert("RGB") [+ transforms.Resize((224, 224))] + Normalize(ImageNet)
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Dataset -------------------------------------------------------------------
# Ricostruiti dagli script di training (verificati sulla Tesla, 05/10/2026):
#   train_ham10000.py  -> legge data/ham10000/images (raw 600x450) tramite
#                         ham10000_prepared.csv; val_tf = DermalAugmentor(False)
#                         = to_tensor + Normalize, SENZA resize ne' DullRazor.
#   train_bcn20000.py  -> legge data/bcn20000/train (512x512); val_tf =
#                         Resize(224) + Normalize, SENZA DullRazor.
#   scripts/train/*_raw.py -> stesse immagini, Resize(224), solo flip;
#                         fold con StratifiedKFold(5, shuffle, seed 42).
#
# input_dir:   immagini lette dal training (cio' che il modello "vede")
# input_size:  224 = Resize((224,224)) come nel val_tf; None = risoluzione nativa
# display_dir: immagini mostrate al clinico nel primo pannello
# input_desc:  descrizione dell'input riportata a pie' di figura
DATASETS = {
    "bcn": {
        "name": "BCN20000",
        "csv": "data/bcn20000/train.csv",
        "input_dir": "data/bcn20000/train",
        "input_size": 224,
        "display_dir": "data/bcn20000/train",
        "input_desc": "immagine originale 512x512 ridimensionata a 224x224, senza DullRazor",
        "cnn_ckpt": "outputs/checkpoints/best_cnn_fold{fold}.pth",
        "hybrid_ckpt": "outputs/checkpoints/best_hybrid_fold{fold}.pth",
        "folds": "stratified_kfold",  # ricostruiti come in train_bcn20000.py
        "kfold_seed": 42,
        "default_fold": 1,  # = fold dei checkpoint globali *_bcn20000.pth
    },
    "ham": {
        "name": "HAM10000",
        "csv": "data/ham10000/ham10000_prepared.csv",
        "input_dir": "data/ham10000/images",
        "input_size": None,  # DermalAugmentor non ridimensiona: 600x450 nativo
        "display_dir": "data/ham10000/images",
        "input_desc": "immagine originale a risoluzione nativa (600x450), senza DullRazor",
        # ResNet-50: esiste solo il checkpoint globale = fold 4 (PR-AUC massima
        # in outputs/metrics_ham10000_cnn.csv)
        "cnn_ckpt": {4: "outputs/checkpoints/best_cnn_ham10000.pth"},
        "hybrid_ckpt": "outputs/checkpoints/best_hybrid_ham10000_fold{fold}.pth",
        "folds": "column",  # colonna 'fold' (1..5) del CSV
        "default_fold": 4,  # unico fold con entrambi i modelli in validazione
    },
    "bcn_raw": {
        "name": "BCN20000 (modelli raw)",
        "csv": "data/bcn20000/train.csv",
        "input_dir": "data/bcn20000/train",
        "input_size": 224,
        "display_dir": "data/bcn20000/train",
        "input_desc": "immagine originale 512x512 ridimensionata a 224x224",
        "cnn_ckpt": "outputs/checkpoints/raw/best_BCN_raw_cnn_fold{fold}.pth",
        "hybrid_ckpt": "outputs/checkpoints/raw/best_BCN_raw_hybrid_fold{fold}.pth",
        "folds": "stratified_kfold",
        "kfold_seed": 42,
        "default_fold": 1,
    },
    "ham_raw": {
        "name": "HAM10000 (modelli raw)",
        "csv": "data/ham10000/HAM10000_metadata.csv",
        "input_dir": "data/ham10000/images",
        "input_size": 224,
        "display_dir": "data/ham10000/images",
        "input_desc": "immagine originale 600x450 ridimensionata a 224x224",
        "cnn_ckpt": "outputs/checkpoints/raw/best_HAM_raw_cnn_fold{fold}.pth",
        "hybrid_ckpt": "outputs/checkpoints/raw/best_HAM_raw_hybrid_fold{fold}.pth",
        "folds": "stratified_kfold",  # come lo script raw (non raggruppato per lesione)
        "kfold_seed": 42,
        "default_fold": 1,
    },
}

# Risoluzione canonica per segmentazione e descrittori (uguale per tutti i
# dataset, cosi' soglie e kernel restano confrontabili)
FEATURE_SIZE = 224

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


def resolve_checkpoint(spec, fold):
  """Percorso del checkpoint per un fold, oppure None se non disponibile."""
  if isinstance(spec, dict):
    return spec.get(fold)
  if "{fold}" in spec:
    return spec.format(fold=fold)
  return spec


def available_folds(spec):
  return sorted(spec) if isinstance(spec, dict) else [1, 2, 3, 4, 5]
