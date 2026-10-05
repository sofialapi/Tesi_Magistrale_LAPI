"""Ricostruzione di fold, processor e modelli; estrazione degli embedding visivi.

Tutto replica src/training/train.py: stesso CSV, stesso StratifiedKFold
(seed 42), ClinicalMetadataProcessor ri-addestrato sul fold di training,
checkpoint best_{case}_fold{k}.pth, trasformazione di validazione del Dataset.
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.multimodal_classifier import DermalClassifier  # noqa: E402
from src.preprocessing.dataset_manager import (  # noqa: E402
    ClinicalMetadataProcessor,
    DermalMultimodalDataset,
)
from XAI_SHAP import shap_config as cfg  # noqa: E402


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(cfg.CSV_PATH, low_memory=False)


def iter_folds(df: pd.DataFrame):
    """Genera (fold 1-based, train_idx, val_idx) identici a train.py."""
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SPLIT_SEED)
    for k, (tr, va) in enumerate(skf.split(df["isic_id"].values, df["target"].values), 1):
        yield k, tr, va


def fit_processor(df_train: pd.DataFrame):
    """Processor del fold (stampe del filtro VIF soppresse) + nomi delle colonne."""
    proc = ClinicalMetadataProcessor()
    with contextlib.redirect_stdout(io.StringIO()):
        X_train = proc.fit_transform(df_train)
    names = list(proc.selected_num_cols) + list(proc.encoder.get_feature_names_out(proc.cat_cols))
    return proc, X_train.astype(np.float32), names


def load_model(case: str, fold: int, clinical_dim: int, device) -> DermalClassifier:
    path = os.path.join(cfg.CKPT_DIR, f"best_{case}_fold{fold}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = DermalClassifier(
        backbone_type=cfg.CASES[case]["backbone"],
        mode="multimodal",
        clinical_dim=clinical_dim,
        pretrained=False,
    )
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def extract_embeddings(model, image_ids, labels, clinical, device, cache_path=None):
    """Embedding visivi (N, D) e logit completi del modello (N, 2) per un insieme di campioni.

    Usa DermalMultimodalDataset con is_training=False: stessa trasformazione
    della validazione in train.py. Il risultato è salvato in cache (.npz).
    """
    image_ids = np.asarray(image_ids)
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if np.array_equal(data["ids"], image_ids):
            return data["emb"], data["logits"]

    ds = DermalMultimodalDataset(
        image_ids=image_ids,
        labels=np.asarray(labels),
        image_dir=cfg.IMAGE_DIR,
        clinical_matrix=np.asarray(clinical, dtype=np.float32),
        mode="multimodal",
        is_training=False,
    )
    loader = DataLoader(ds, batch_size=cfg.IMG_BATCH, shuffle=False, num_workers=cfg.NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())
    embs, logits = [], []
    for img, clin, _ in loader:
        img, clin = img.to(device), clin.to(device)
        emb = model.visual_backbone(img)
        out = model.classifier(torch.cat((emb, model.tabular_encoder(clin)), dim=1))
        embs.append(emb.cpu().numpy())
        logits.append(out.cpu().numpy())
    emb, logit = np.concatenate(embs), np.concatenate(logits)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, ids=image_ids, emb=emb, logits=logit)
    return emb, logit


class HeadFunction:
    """Testa di fusione scomposta per un calcolo efficiente.

    Il primo layer della testa è lineare:  W1 [e ; h] + b1 = (W1_e e + b1) + W1_h h.
    Il contesto visivo c = W1_e e + b1 (128 valori) si calcola una volta per
    immagine; per ogni coalizione resta da valutare solo TabularMLP e il resto
    della testa. Il risultato è identico al forward completo (modello in eval).
    Output spiegato: log-odds della classe maligna, z1 - z0 (= logit di softmax[:, 1]).
    """

    def __init__(self, model: DermalClassifier, device, max_rows: int = 131072):
        self.model, self.device, self.max_rows = model, device, max_rows
        lin = model.classifier[0]
        vis_dim = model.visual_backbone.feature_dim
        self.W_img = lin.weight[:, :vis_dim].detach()
        self.W_tab = lin.weight[:, vis_dim:].detach()
        self.bias = lin.bias.detach()
        self.rest = model.classifier[1:]

    @torch.no_grad()
    def project(self, emb: np.ndarray) -> np.ndarray:
        e = torch.as_tensor(emb, dtype=torch.float32, device=self.device)
        return (e @ self.W_img.T + self.bias).cpu().numpy()

    @torch.no_grad()
    def predict(self, ctx_rows: np.ndarray, tab_rows: np.ndarray) -> np.ndarray:
        out = np.empty(len(tab_rows), dtype=np.float64)
        for s in range(0, len(tab_rows), self.max_rows):
            c = torch.as_tensor(np.ascontiguousarray(ctx_rows[s:s + self.max_rows]), dtype=torch.float32, device=self.device)
            t = torch.as_tensor(np.ascontiguousarray(tab_rows[s:s + self.max_rows]), dtype=torch.float32, device=self.device)
            z = self.rest(c + self.model.tabular_encoder(t) @ self.W_tab.T)
            out[s:s + len(t)] = (z[:, 1] - z[:, 0]).double().cpu().numpy()
        return out


def best_epoch_prauc(case: str):
    """PR-AUC della riga con validation loss minima per fold, dal CSV delle metriche."""
    name = {"cnn_multimodal": "metrics_multimodal.csv", "hybrid_multimodal": "metrics_hybrid_multimodal.csv"}[case]
    path = os.path.join(cfg.ROOT, "outputs", name)
    if not os.path.exists(path):
        return {}
    m = pd.read_csv(path)
    best = m.loc[m.groupby("fold")["val_loss"].idxmin()]
    return dict(zip(best["fold"].astype(int), best["pr_auc"].astype(float)))
