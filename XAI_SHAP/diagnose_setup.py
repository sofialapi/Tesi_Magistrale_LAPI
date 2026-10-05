"""Diagnosi preliminare per l'analisi SHAP su ISIC 2024 (sola lettura)."""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.preprocessing.dataset_manager import ClinicalMetadataProcessor  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "outputs", "checkpoints")
CSV_PATH = os.path.join(ROOT, "data", "raw", "ISIC", "metadata_isic_subset.csv")
CASES = ["cnn_only", "hybrid_only", "cnn_multimodal", "hybrid_multimodal"]


def ts(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------------------
section("1. CHECKPOINT ISIC 2024")
ckpt_dims = {}
for case in CASES:
    for fold in range(1, 6):
        p = os.path.join(CKPT_DIR, f"best_{case}_fold{fold}.pth")
        if not os.path.exists(p):
            print(f"  [MANCA] {os.path.basename(p)}")
            continue
        obj = torch.load(p, map_location="cpu")
        kind = type(obj).__name__
        sd = obj
        if isinstance(obj, dict) and "model_state_dict" in obj:
            sd = obj["model_state_dict"]
            kind += " (wrapper model_state_dict)"
        extra = ""
        if isinstance(sd, dict):
            extra += f"{len(sd)} tensori"
            if "tabular_encoder.net.0.weight" in sd:
                d = sd["tabular_encoder.net.0.weight"].shape[1]
                ckpt_dims[(case, fold)] = d
                extra += f" | clinical_dim={d}"
            if "classifier.4.weight" in sd:
                extra += f" | n_output={sd['classifier.4.weight'].shape[0]}"
        print(f"  {os.path.basename(p):38s} {ts(p)}  {kind}, {extra}")

others = sorted(
    f for f in os.listdir(CKPT_DIR)
    if f.endswith(".pth") and not any(f.startswith(f"best_{c}_fold") for c in CASES)
)
print("\n  Altri .pth nella cartella (naming diverso):")
for f in others:
    print(f"    {f:40s} {ts(os.path.join(CKPT_DIR, f))}")

# ---------------------------------------------------------------------------
section("2. IL PROCESSOR VIENE SALVATO?")
pattern = re.compile(r"joblib|pickle|\.dump\(|to_pickle|np\.save")
skip = ("venv", "site-packages", ".git", "__pycache__", os.sep + "data")
hits = []
for d, _, files in os.walk(ROOT):
    if any(s in d for s in skip):
        continue
    for f in files:
        fp = os.path.join(d, f)
        if not f.endswith(".py") or os.path.abspath(fp) == os.path.abspath(__file__):
            continue
        with open(fp, errors="ignore") as fh:
            for i, line in enumerate(fh, 1):
                if pattern.search(line):
                    hits.append(f"  {os.path.relpath(fp, ROOT)}:{i}: {line.strip()}")
print("\n".join(hits) if hits else "  Nessuna istruzione di salvataggio trovata nel codice.")

saved = []
for d, _, files in os.walk(os.path.join(ROOT, "outputs")):
    saved += [os.path.join(d, f) for f in files if f.endswith((".pkl", ".joblib", ".npy", ".npz"))]
print("  File serializzati in outputs/: " + (", ".join(os.path.relpath(s, ROOT) for s in saved) or "nessuno"))

# ---------------------------------------------------------------------------
section("3. CSV DEI METADATI")
print(f"  {CSV_PATH}  (modificato: {ts(CSV_PATH)})")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Righe: {len(df)} | Colonne: {df.shape[1]}")
print(f"  Target: {df['target'].value_counts().to_dict()}")
proc_tmp = ClinicalMetadataProcessor()
for c in proc_tmp.num_cols:
    s = df[c]
    print(f"  [num] {c:22s} mancanti={s.isna().sum():5d}  min={s.min():.3f}  mediana={s.median():.3f}  max={s.max():.3f}")
for c in proc_tmp.cat_cols:
    vc = df[c].value_counts(dropna=False).to_dict()
    print(f"  [cat] {c:22s} {vc}")

# ---------------------------------------------------------------------------
section("4. RICOSTRUZIONE DEL PROCESSOR PER FOLD (stesso split di train.py)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(skf.split(df["isic_id"].values, df["target"].values), 1):
    proc = ClinicalMetadataProcessor()
    X_tr = proc.fit_transform(df.iloc[tr].copy())
    names = list(proc.selected_num_cols) + list(proc.encoder.get_feature_names_out(proc.cat_cols))
    checks = []
    for case in ("cnn_multimodal", "hybrid_multimodal"):
        d = ckpt_dims.get((case, fold))
        if d is None:
            checks.append(f"{case}: checkpoint assente")
        else:
            checks.append(f"{case}: {'OK' if d == X_tr.shape[1] else 'DIVERSO (' + str(d) + ')'}")
    print(f"  Fold {fold}: train={len(tr)} val={len(va)} (maligni val={int(df['target'].values[va].sum())})")
    print(f"           dim={X_tr.shape[1]} | numeriche={proc.selected_num_cols}")
    print(f"           feature: {names}")
    print(f"           confronto checkpoint -> {' | '.join(checks)}")

# ---------------------------------------------------------------------------
section("5. CSV DELLE METRICHE")
out_dir = os.path.join(ROOT, "outputs")
for f in sorted(os.listdir(out_dir)):
    if f.startswith("metrics_") and f.endswith(".csv"):
        p = os.path.join(out_dir, f)
        m = pd.read_csv(p)
        print(f"\n  {f}  (modificato: {ts(p)})  shape={m.shape}")
        print("  colonne:", list(m.columns))
        print(m.head(3).to_string(index=False))
