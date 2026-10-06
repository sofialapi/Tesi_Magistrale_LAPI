"""Rigenera con stile uniforme tutte le figure di addestramento (Cap. 4 e 5).

Uso, dalla root del progetto:   python make_uniform_plots.py

Le figure vanno in uniform_plots/ con la stessa struttura delle cartelle LaTeX:
    uniform_plots/cap4/BCN, cap4/HAM, cap4/VGG, cap5
piu' le figure non incluse nella tesi in uniform_plots/raw e uniform_plots/vgg.
Le figure esistenti non vengono modificate.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps

OUT = "uniform_plots"
METRICS3 = ["pr_auc", "recall", "mcc"]

# Cap. 5: le curve per epoca sono mostrate solo dove almeno MIN_FOLDS_CAP5 fold
# su 5 sono ancora in addestramento (early stopping).
MIN_FOLDS_CAP5 = 3

# Tabella 5.4: l'MCC non e' salvato nei CSV per epoca del Cap. 5, quindi nel
# barchart viene preso da qui (senza barra d'errore). PR-AUC e Sensibilita'
# vengono invece ricalcolate dai CSV e confrontate con questi valori.
TAB_CAP5 = {
    "Caso 1 – ResNet-50": {"pr_auc": 0.4151, "recall": 0.4581, "mcc": 0.4168},
    "Caso 2 – MobileViT-S": {"pr_auc": 0.4705, "recall": 0.4070, "mcc": 0.4531},
    "Caso 3 – ResNet-50 + clinici": {"pr_auc": 0.4876, "recall": 0.4964, "mcc": 0.4814},
    "Caso 4 – MobileViT-S + clinici": {"pr_auc": 0.4671, "recall": 0.3892, "mcc": 0.4506},
}


def _exists(files):
    missing = [p for p in files.values() if not os.path.exists(p)]
    for p in missing:
        print(f"  [SALTATO] CSV mancante: {p}")
    return not missing


# ---------------------------------------------------------------------------
# Capitolo 4: le quattro figure standard per un confronto tra due modelli
# ---------------------------------------------------------------------------
def two_model_set(files, out_dir, prefix, title, loss_label="BCE pesata"):
    print(f"\n=== {title} ===")
    if not _exists(files):
        return
    models = list(files)
    df = ps.load_runs(files)
    best = ps.best_per_fold(df, "max_pr_auc")
    ps.print_summary(best, models, METRICS3 + ["val_loss"])

    # 1. Training e validation loss
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.5))
    for ax, (col, name) in zip(axes, [("train_loss", "Training loss"),
                                      ("val_loss", "Validation loss")]):
        ps.plot_curves(ax, df, col, models)
        ax.set_title(name)
        ax.set_ylabel(loss_label)
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(models), ncol=len(models))
    ps.save(fig, os.path.join(out_dir, f"{prefix}_loss_comparison.png"))

    # 2. Metriche per epoca
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.3))
    for ax, col in zip(axes, METRICS3):
        ps.plot_curves(ax, df, col, models)
        ax.set_title(ps.METRIC_LABELS[col])
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(models), ncol=len(models))
    ps.save(fig, os.path.join(out_dir, f"{prefix}_metrics_progression.png"))

    # 3. Distribuzione sui fold
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.4))
    for ax, col in zip(axes, METRICS3):
        ps.box_panel(ax, best, col, models)
        ax.set_title(ps.METRIC_LABELS[col])
    fig.tight_layout()
    ps.save(fig, os.path.join(out_dir, f"{prefix}_boxplot_comparison.png"))

    # 4. Medie con deviazione standard
    stats = {m: {c: (best.loc[best.Model == m, c].mean(), best.loc[best.Model == m, c].std())
                 for c in METRICS3} for m in models}
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.6))
    ps.bar_panel(ax, stats, models, METRICS3)
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(models, kind="patch"), ncol=len(models))
    ps.save(fig, os.path.join(out_dir, f"{prefix}_global_barchart.png"))


# ---------------------------------------------------------------------------
# Capitolo 4: scelta della baseline (VGG-16 vs ResNet-50, MobileViT-S come riferimento)
# ---------------------------------------------------------------------------
def selezione_backbone():
    print("\n=== Cap. 4 - Selezione backbone (Tab. 4.1) ===")
    runs = {
        "BCN20000": {
            "VGG-16": "outputs/vgg/metrics/metrics_vgg_bcn20000_cnn.csv",
            "ResNet-50": "outputs/metrics_bcn_cnn.csv",
            "MobileViT-S": "outputs/metrics_bcn_hybrid.csv",
        },
        "HAM10000": {
            "VGG-16": "outputs/vgg/metrics/metrics_vgg_ham10000_cnn.csv",
            "ResNet-50": "outputs/metrics_ham10000_cnn.csv",
            "MobileViT-S": "outputs/metrics_ham10000_hybrid.csv",
        },
    }
    if not all(_exists(f) for f in runs.values()):
        return
    models = ["VGG-16", "ResNet-50", "MobileViT-S"]
    out_dir = os.path.join(OUT, "cap4", "VGG")
    hist = {ds: ps.load_runs(f) for ds, f in runs.items()}
    best = {ds: ps.best_per_fold(h, "max_pr_auc") for ds, h in hist.items()}
    for ds in runs:
        print(f" [{ds}]")
        ps.print_summary(best[ds], models, METRICS3)

    # 1. Valori per fold, linee che collegano lo stesso fold.
    #    Metrica nel titolo (riga in alto), dataset come etichetta di riga,
    #    modelli identificati dalla legenda (stesso ordine da sinistra a destra).
    fig, axes = plt.subplots(2, 3, figsize=(ps.FULL_W, 4.3))
    for i, ds in enumerate(runs):
        sub = best[ds]
        for j, col in enumerate(METRICS3):
            ax = axes[i, j]
            for _, g in sub.groupby("fold"):
                g = g.set_index("Model").reindex(models)
                ax.plot(range(len(models)), g[col].values, color=ps.LIGHT, lw=0.8, zorder=1)
            for k, m in enumerate(models):
                vals = sub.loc[sub.Model == m, col].to_numpy()
                ax.scatter(np.full(len(vals), k), vals, s=20,
                           color=ps.MODEL_STYLE[m]["color"], edgecolor=ps.INK, lw=0.4, zorder=2)
                ax.scatter(k, vals.mean(), marker="D", s=30, facecolor="white",
                           edgecolor=ps.INK, lw=1.0, zorder=3)
            ax.set_xticks([])
            ax.set_xlim(-0.4, len(models) - 0.6)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(ps.METRIC_LABELS[col])
            if j == 0:
                ax.set_ylabel(ds, fontweight="bold")
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(models, kind="point") + [ps.mean_handle()], ncol=4)
    ps.save(fig, os.path.join(out_dir, "selezione_backbone_per_fold.png"))

    # 2. Loss VGG-16 vs ResNet-50
    cnn = ["VGG-16", "ResNet-50"]
    fig, axes = plt.subplots(2, 2, figsize=(ps.FULL_W, 4.4))
    for i, ds in enumerate(runs):
        h = hist[ds][hist[ds].Model.isin(cnn)]
        for j, (col, name) in enumerate([("train_loss", "Training loss"),
                                         ("val_loss", "Validation loss")]):
            ax = axes[i, j]
            ps.plot_curves(ax, h, col, cnn)
            ax.set_title(f"{name} – {ds}")
            ax.set_ylabel("BCE pesata")
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(cnn), ncol=2)
    ps.save(fig, os.path.join(out_dir, "selezione_backbone_loss.png"))


# ---------------------------------------------------------------------------
# Capitolo 5: quattro configurazioni su ISIC 2024
# ---------------------------------------------------------------------------
def cap5():
    print("\n=== Cap. 5 - ISIC 2024 (Tab. 5.1-5.4) ===")
    files = {
        "Caso 1 – ResNet-50": "outputs/metrics_cnn_only.csv",
        "Caso 2 – MobileViT-S": "outputs/metrics_hybrid_only.csv",
        "Caso 3 – ResNet-50 + clinici": "outputs/metrics_multimodal.csv",
        "Caso 4 – MobileViT-S + clinici": "outputs/metrics_hybrid_multimodal.csv",
    }
    if not _exists(files):
        return
    models = list(files)
    ticks = ["Caso 1\nResNet-50", "Caso 2\nMobileViT-S",
             "Caso 3\nResNet-50\n+ clinici", "Caso 4\nMobileViT-S\n+ clinici"]
    out_dir = os.path.join(OUT, "cap5")
    df = ps.load_runs(files)
    # Stesso checkpoint di train.py: epoca con validation loss minima per fold
    best = ps.best_per_fold(df, "min_val_loss")

    print("  Controllo rispetto alla Tabella 5.4 (CSV, epoca a val loss minima):")
    for m in models:
        b = best[best.Model == m]
        line = []
        for c in ["pr_auc", "recall"]:
            diff = b[c].mean() - TAB_CAP5[m][c]
            flag = "" if abs(diff) < 0.002 else "  <-- DIVERSO"
            line.append(f"{c} {b[c].mean():.4f} ± {b[c].std():.4f} (tab {TAB_CAP5[m][c]:.4f}){flag}")
        print(f"  {m:<32} | " + " | ".join(line) + f" | epoche per fold {b.epoch.tolist()}")

    print(f"  Ultima epoca mostrata nelle curve (almeno {MIN_FOLDS_CAP5} fold attivi):")
    for m in models:
        n = df[df.Model == m].groupby("epoch")["fold"].nunique()
        print(f"    {m:<32} epoca {int(n[n >= MIN_FOLDS_CAP5].index.max())}"
              f" (ultima epoca raggiunta da un fold: {int(n.index.max())})")

    band_alpha = 0.10  # 4 curve: banda piu' leggera; band=False per toglierla

    # 1. Focal loss
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.5))
    for ax, (col, name) in zip(axes, [("train_loss", "Training loss"),
                                      ("val_loss", "Validation loss")]):
        ps.plot_curves(ax, df, col, models, band_alpha=band_alpha, min_folds=MIN_FOLDS_CAP5)
        ax.set_title(name)
        ax.set_ylabel("Focal Loss")
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(models), ncol=2)
    ps.save(fig, os.path.join(out_dir, "loss_curves_comparison.png"))

    # 2. PR-AUC e Sensibilita' per epoca (MCC non disponibile nei CSV)
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.5))
    for ax, col in zip(axes, ["pr_auc", "recall"]):
        ps.plot_curves(ax, df, col, models, band_alpha=band_alpha, min_folds=MIN_FOLDS_CAP5)
        ax.set_title(ps.METRIC_LABELS[col])
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(models), ncol=2)
    ps.save(fig, os.path.join(out_dir, "metrics_progression_comparison.png"))

    # 3. Distribuzione sui fold
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.8))
    for ax, col in zip(axes, ["pr_auc", "recall"]):
        ps.box_panel(ax, best, col, models, tick_labels=ticks, tick_fontsize=7.5)
        ax.set_title(ps.METRIC_LABELS[col])
    fig.tight_layout()
    ps.save(fig, os.path.join(out_dir, "boxplot_cross_validation.png"))

    # 4. Medie: PR-AUC e Sensibilita' dai CSV, MCC dalla Tabella 5.4
    stats = {}
    for m in models:
        b = best[best.Model == m]
        stats[m] = {"pr_auc": (b.pr_auc.mean(), b.pr_auc.std()),
                    "recall": (b.recall.mean(), b.recall.std()),
                    "mcc": (TAB_CAP5[m]["mcc"], np.nan)}
    fig, ax = plt.subplots(figsize=(ps.FULL_W, 2.8))
    ps.bar_panel(ax, stats, models, METRICS3)
    fig.tight_layout()
    ps.top_legend(fig, ps.model_handles(models, kind="patch"), ncol=2)
    ps.save(fig, os.path.join(out_dir, "global_metrics_barchart.png"))


def main():
    ps.apply_style()

    # Figure della tesi
    two_model_set({"ResNet-50": "outputs/metrics_bcn_cnn.csv",
                   "MobileViT-S": "outputs/metrics_bcn_hybrid.csv"},
                  os.path.join(OUT, "cap4", "BCN"), "bcn", "Cap. 4 - BCN20000 (Tab. 4.2)")
    two_model_set({"ResNet-50": "outputs/metrics_ham10000_cnn.csv",
                   "MobileViT-S": "outputs/metrics_ham10000_hybrid.csv"},
                  os.path.join(OUT, "cap4", "HAM"), "ham", "Cap. 4 - HAM10000 (Tab. 4.3)")
    selezione_backbone()
    cap5()

    # Figure di supporto, non incluse nella tesi
    two_model_set({"ResNet-50": "outputs/raw/metrics_BCN_raw_cnn.csv",
                   "MobileViT-S": "outputs/raw/metrics_BCN_raw_hybrid.csv"},
                  os.path.join(OUT, "raw", "bcn20000"), "bcn_raw", "Raw - BCN20000 (Tab. 4.4)")
    two_model_set({"ResNet-50": "outputs/raw/metrics_HAM_raw_cnn.csv",
                   "MobileViT-S": "outputs/raw/metrics_HAM_raw_hybrid.csv"},
                  os.path.join(OUT, "raw", "ham10000"), "ham_raw", "Raw - HAM10000 (Tab. 4.4)")
    two_model_set({"VGG-16": "outputs/vgg/metrics/metrics_vgg_bcn20000_cnn.csv",
                   "MobileViT-S": "outputs/vgg/metrics/metrics_vgg_bcn20000_hybrid.csv"},
                  os.path.join(OUT, "vgg"), "bcn_vgg", "VGG run - BCN20000")
    two_model_set({"VGG-16": "outputs/vgg/metrics/metrics_vgg_ham10000_cnn.csv",
                   "MobileViT-S": "outputs/vgg/metrics/metrics_vgg_ham10000_hybrid.csv"},
                  os.path.join(OUT, "vgg"), "ham_vgg", "VGG run - HAM10000")

    print(f"\nFatto. Figure in {OUT}/")


if __name__ == "__main__":
    main()
