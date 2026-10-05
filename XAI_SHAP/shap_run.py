"""Calcolo dei valori SHAP per i Casi 3 e 4 su ISIC 2024 (5 fold, out-of-fold).

Uso (dalla radice del progetto):
    python -m XAI_SHAP.shap_run                       # entrambi i casi, tutti i fold
    python -m XAI_SHAP.shap_run --cases cnn_multimodal --folds 1

Per ogni lesione di validazione vengono calcolati:
  * phi_<variabile>: Shapley esatti delle 6 variabili cliniche (one-hot raggruppate,
    2^6 = 64 coalizioni), con l'immagine fissata; somma = logit - base_tab.
  * phi_img, phi_tab: Shapley a due giocatori immagine vs clinica.
  * PR-AUC del modello completo e con ciascuna modalità marginalizzata.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from XAI_SHAP import shap_config as cfg
from XAI_SHAP.shap_core import build_groups, modality_shapley, shapley_matrix, tabular_coalition_values
from XAI_SHAP.shap_data import (
    HeadFunction,
    best_epoch_prauc,
    extract_embeddings,
    fit_processor,
    iter_folds,
    load_metadata,
    load_model,
)


def run_case(case, df, folds, device, bg_size, use_cache):
    print(f"\n=== {cfg.CASES[case]['label']} ===")
    csv_prauc = best_epoch_prauc(case)
    records, fold_rows = [], []

    for fold, tr, va in iter_folds(df):
        if fold not in folds:
            continue
        t0 = time.time()
        df_tr, df_va = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
        proc, X_tr, names = fit_processor(df_tr)
        X_va = proc.transform(df_va).astype(np.float32)
        groups = build_groups(names, cfg.VAR_ORDER)

        rng = np.random.default_rng(cfg.BACKGROUND_SEED + fold)
        bg_idx = np.sort(rng.choice(len(df_tr), size=min(bg_size, len(df_tr)), replace=False))
        X_bg = X_tr[bg_idx]

        model = load_model(case, fold, X_tr.shape[1], device)
        head = HeadFunction(model, device)

        cache = (lambda split: os.path.join(cfg.CACHE_DIR, f"{case}_fold{fold}_{split}.npz")) if use_cache else (lambda s: None)
        emb_va, full_logits = extract_embeddings(model, df_va["isic_id"].values, df_va["target"].values, X_va, device, cache("val"))
        emb_bg, _ = extract_embeddings(model, df_tr["isic_id"].values[bg_idx], df_tr["target"].values[bg_idx], X_bg, device, cache(f"bg{bg_size}"))

        ctx_va, ctx_bg = head.project(emb_va), head.project(emb_bg)

        # Controllo: la testa scomposta deve riprodurre il forward completo.
        logit_model = full_logits[:, 1] - full_logits[:, 0]
        logit_head = head.predict(ctx_va, X_va)
        proj_err = float(np.abs(logit_head - logit_model).max())

        # 1) Shapley esatti sulle 6 variabili cliniche (immagine fissata).
        values, masks = tabular_coalition_values(head.predict, ctx_va, X_va, X_bg, groups, chunk=cfg.SAMPLE_CHUNK)
        phi = values @ shapley_matrix(masks).T
        v_full, v_img_only = values[:, -1], values[:, 0]
        eff_err = float(np.abs(phi.sum(axis=1) - (v_full - v_img_only)).max())

        # 2) Shapley immagine vs clinica.
        n, b = len(X_va), len(X_bg)
        v_tab_only = np.empty(n)
        step = max(1, cfg.SAMPLE_CHUNK * 64)
        for s in range(0, n, step):
            t = X_va[s:s + step]
            ctx_rows = np.broadcast_to(ctx_bg[None], (len(t), b, ctx_bg.shape[1])).reshape(-1, ctx_bg.shape[1])
            tab_rows = np.repeat(t, b, axis=0)
            v_tab_only[s:s + len(t)] = head.predict(ctx_rows, tab_rows).reshape(len(t), b).mean(axis=1)
        v_none = float(head.predict(ctx_bg, X_bg).mean())
        phi_img, phi_tab = modality_shapley(v_full, v_img_only, v_tab_only, v_none)

        y = df_va["target"].values
        row = {
            "fold": fold, "n_val": n, "n_pos": int(y.sum()),
            "prauc_csv_best_epoch": csv_prauc.get(fold, np.nan),
            "prauc_full": average_precision_score(y, v_full),
            "prauc_tab_marginalized": average_precision_score(y, v_img_only),
            "prauc_img_marginalized": average_precision_score(y, v_tab_only),
            "v_none": v_none,
            "max_err_projection": proj_err,
            "max_err_efficiency": eff_err,
            "seconds": round(time.time() - t0, 1),
        }
        fold_rows.append(row)
        print(f"  Fold {fold}: PR-AUC {row['prauc_full']:.4f} (CSV {row['prauc_csv_best_epoch']:.4f}) | "
              f"senza clinica {row['prauc_tab_marginalized']:.4f} | senza immagine {row['prauc_img_marginalized']:.4f} | "
              f"err proiezione {proj_err:.1e} | err efficienza {eff_err:.1e} | {row['seconds']} s")

        rec = pd.DataFrame({
            "isic_id": df_va["isic_id"].values, "fold": fold, "target": y,
            "logit": v_full, "prob": 1.0 / (1.0 + np.exp(-v_full)),
            "base_tab": v_img_only, "v_tab_only": v_tab_only, "v_none": v_none,
            "phi_img": phi_img, "phi_tab": phi_tab,
        })
        for g, var in enumerate(cfg.VAR_ORDER):
            rec[f"phi_{var}"] = phi[:, g]
            rec[f"val_{var}"] = df_va[var].values
        records.append(rec)

        del model, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = pd.concat(records, ignore_index=True)
    folds_df = pd.DataFrame(fold_rows)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(cfg.OUT_DIR, f"shap_values_{case}.csv"), index=False)
    folds_df.to_csv(os.path.join(cfg.OUT_DIR, f"fold_summary_{case}.csv"), index=False)
    return out, folds_df


def main():
    p = argparse.ArgumentParser(description="SHAP sui modelli multimodali ISIC 2024")
    p.add_argument("--cases", nargs="+", default=list(cfg.CASES), choices=list(cfg.CASES))
    p.add_argument("--folds", nargs="+", type=int, default=list(range(1, cfg.N_FOLDS + 1)))
    p.add_argument("--background_size", type=int, default=cfg.BACKGROUND_SIZE)
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    print(f"Device: {args.device} | background: {args.background_size} | fold: {args.folds}")
    df = load_metadata()
    meta = {"background_size": args.background_size, "background_seed": cfg.BACKGROUND_SEED,
            "folds": args.folds, "variables": cfg.VAR_ORDER, "output": "log-odds z1 - z0"}
    for case in args.cases:
        _, folds_df = run_case(case, df, set(args.folds), args.device, args.background_size, not args.no_cache)
        meta[case] = folds_df.drop(columns=["seconds"]).mean(numeric_only=True).to_dict()
    with open(os.path.join(cfg.OUT_DIR, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print(f"\nRisultati in {cfg.OUT_DIR}. Figure: python -m XAI_SHAP.shap_plots")


if __name__ == "__main__":
    main()
