import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configurazione grafica coerente con tesi
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})
sns.set_theme(style="whitegrid")

# Palette colori: Porpora/Rosso (#8c564b o #d62728) per VGG-16, Arancione (#ff7f0e) per MobileViT-S
palette = {
    "VGG-16 (CNN)": "#d62728",
    "MobileViT-S (Hybrid)": "#ff7f0e"
}
model_order = ["VGG-16 (CNN)", "MobileViT-S (Hybrid)"]

out_dir = "figure/vgg"
os.makedirs(out_dir, exist_ok=True)

def process_benchmark(dataset_name, cnn_csv, hyb_csv, prefix):
    print(f"\n{'='*25} ANALISI METRICHE: {dataset_name} {'='*25}")
    if not os.path.exists(cnn_csv) or not os.path.exists(hyb_csv):
        print(f"[ERRORE] File mancanti per {dataset_name}: {cnn_csv} o {hyb_csv}")
        return

    df_cnn = pd.read_csv(cnn_csv)
    df_hyb = pd.read_csv(hyb_csv)
    df_cnn["Model"] = "VGG-16 (CNN)"
    df_hyb["Model"] = "MobileViT-S (Hybrid)"

    df_all = pd.concat([df_cnn, df_hyb], ignore_index=True)

    # Selezione del miglior checkpoint per fold guidato da PR-AUC
    best_per_fold = (
        df_all.sort_values(by=["Model", "fold", "pr_auc"], ascending=False)
        .groupby(["Model", "fold"])
        .first()
        .reset_index()
    )

    # 1. Stampa Tabella Sintetica per Tesi LaTeX
    summary = best_per_fold.groupby("Model")[["pr_auc", "recall", "mcc", "val_loss"]].agg(["mean", "std"])
    print("\n--- RISULTATI MEDI SUI 5 FOLD (best PR-AUC per fold) ---")
    for m in model_order:
        p_mean, p_std = summary.loc[m, ("pr_auc", "mean")], summary.loc[m, ("pr_auc", "std")]
        r_mean, r_std = summary.loc[m, ("recall", "mean")], summary.loc[m, ("recall", "std")]
        m_mean, m_std = summary.loc[m, ("mcc", "mean")], summary.loc[m, ("mcc", "std")]
        vl_mean, vl_std = summary.loc[m, ("val_loss", "mean")], summary.loc[m, ("val_loss", "std")]
        print(f"{m:<22} | PR-AUC: {p_mean:.4f} ± {p_std:.4f} | Recall: {r_mean*100:.2f}% ± {r_std*100:.2f}% | MCC: {m_mean:.4f} ± {m_std:.4f} | ValLoss: {vl_mean:.4f} ± {vl_std:.4f}")

    # -------------------------------------------------------------------------
    # GRAFICO 1: Curve di Loss (Training vs Validation)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    sns.lineplot(
        data=df_all, x="epoch", y="train_loss", hue="Model",
        hue_order=model_order, palette=palette, ax=axes[0], errorbar="sd", linewidth=2.2
    )
    axes[0].set_title(f"Training Loss Media ({dataset_name} - 5 Fold)")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")

    sns.lineplot(
        data=df_all, x="epoch", y="val_loss", hue="Model",
        hue_order=model_order, palette=palette, ax=axes[1], errorbar="sd", linewidth=2.2
    )
    axes[1].set_title(f"Validation Loss Media ({dataset_name} - 5 Fold)")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")

    plt.tight_layout()
    p1 = os.path.join(out_dir, f"{prefix}_loss_comparison.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {p1}")

    # -------------------------------------------------------------------------
    # GRAFICO 2: Evoluzione Epocale di PR-AUC, Recall e MCC
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metrics_prog = [
        ("pr_auc", "PR-AUC"),
        ("recall", "Recall (Sensibilità)"),
        ("mcc", "Matthews Corr. Coeff. (MCC)")
    ]

    for idx, (col, title) in enumerate(metrics_prog):
        sns.lineplot(
            data=df_all, x="epoch", y=col, hue="Model",
            hue_order=model_order, palette=palette, ax=axes[idx], errorbar=None, linewidth=2.2
        )
        axes[idx].set_title(title, fontweight="bold")
        axes[idx].set_xlabel("Epoca")
        axes[idx].set_ylabel(title)

    plt.suptitle(f"Dinamica delle Metriche lungo le 20 Epoche - {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"{prefix}_metrics_progression.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {p2}")

    # -------------------------------------------------------------------------
    # GRAFICO 3: Boxplot sui 5 Fold (VGG-16 a Sx, MobileViT a Dx)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title) in zip(axes, metrics_prog):
        sns.boxplot(
            data=best_per_fold, x="Model", y=col, order=model_order,
            palette=palette, ax=ax, width=0.4, showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}
        )
        sns.stripplot(
            data=best_per_fold, x="Model", y=col, order=model_order,
            color="black", alpha=0.6, jitter=0.15, size=7, ax=ax
        )
        ax.set_title(f"Distribuzione {title}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(title)

    plt.suptitle(f"Variabilità e Dispersione sui 5 Fold - {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(out_dir, f"{prefix}_boxplot_comparison.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {p3}")

    # -------------------------------------------------------------------------
    # GRAFICO 4: Barchart Riassuntivo con Error Bars
    # -------------------------------------------------------------------------
    labels = ["PR-AUC", "Recall", "MCC"]
    x = np.arange(len(labels))
    width = 0.35

    cnn_means = [summary.loc["VGG-16 (CNN)", (m, "mean")] for m in ["pr_auc", "recall", "mcc"]]
    cnn_stds = [summary.loc["VGG-16 (CNN)", (m, "std")] for m in ["pr_auc", "recall", "mcc"]]
    hyb_means = [summary.loc["MobileViT-S (Hybrid)", (m, "mean")] for m in ["pr_auc", "recall", "mcc"]]
    hyb_stds = [summary.loc["MobileViT-S (Hybrid)", (m, "std")] for m in ["pr_auc", "recall", "mcc"]]

    plt.figure(figsize=(9, 5.5))
    r1 = plt.bar(x - width/2, cnn_means, width, yerr=cnn_stds, capsize=5, label="VGG-16 (CNN)", color=palette["VGG-16 (CNN)"], alpha=0.9)
    r2 = plt.bar(x + width/2, hyb_means, width, yerr=hyb_stds, capsize=5, label="MobileViT-S (Hybrid)", color=palette["MobileViT-S (Hybrid)"], alpha=0.9)

    plt.ylabel("Punteggio Medio")
    plt.title(f"Benchmark Finale {dataset_name}: VGG-16 vs MobileViT-S (5 Fold CV)", fontweight="bold")
    plt.xticks(x, labels, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend(frameon=True)

    for rect in list(r1) + list(r2):
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p4 = os.path.join(out_dir, f"{prefix}_global_barchart.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[OK] Salvato: {p4}")

def main():
    # 1. HAM10000
    process_benchmark(
        dataset_name="HAM10000",
        cnn_csv="outputs/vgg/metrics/metrics_vgg_ham10000_cnn.csv",
        hyb_csv="outputs/vgg/metrics/metrics_vgg_ham10000_hybrid.csv",
        prefix="ham_vgg"
    )

    # 2. BCN20000
    process_benchmark(
        dataset_name="BCN20000",
        cnn_csv="outputs/vgg/metrics/metrics_vgg_bcn20000_cnn.csv",
        hyb_csv="outputs/vgg/metrics/metrics_vgg_bcn20000_hybrid.csv",
        prefix="bcn_vgg"
    )

if __name__ == "__main__":
    main()
