import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Stile tipografico coerente con tesi magistrale / pubblicazione scientifica
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})
sns.set_theme(style="whitegrid")

# 1. Caricamento dei file CSV estratti da Tesla
files = {
    "CNN (ResNet-50)": "outputs/metrics_bcn_cnn.csv",
    "Hybrid (MobileViT-S)": "outputs/metrics_bcn_hybrid.csv",
}

palette = {
    "CNN (ResNet-50)": "#4C72B0",  # Blu istituzionale
    "Hybrid (MobileViT-S)": "#C44E52",  # Rosso carminio / contrasto
}

dfs = []
for model_name, file_path in files.items():
    if os.path.exists(file_path):
        df_temp = pd.read_csv(file_path)
        df_temp["Model"] = model_name
        dfs.append(df_temp)
    else:
        print(f"[ATTENZIONE] File non trovato: {file_path}")

if not dfs:
    raise FileNotFoundError(
        "Nessun file CSV trovato. Assicurati di aver fatto 'git pull origin main'."
    )

df_all = pd.concat(dfs, ignore_index=True)
os.makedirs("outputs/plots/bcn20000", exist_ok=True)

# -------------------------------------------------------------------------
# GRAFICO 1: Curve di Loss (Training vs Validation)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Training Loss
sns.lineplot(
    data=df_all,
    x="epoch",
    y="train_loss",
    hue="Model",
    palette=palette,
    ax=axes[0],
    errorbar="sd",
    linewidth=2.2,
)
axes[0].set_title("Training Loss Media (BCN20000 - 5 Fold)")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("Binary Cross-Entropy Loss")

# Validation Loss
sns.lineplot(
    data=df_all,
    x="epoch",
    y="val_loss",
    hue="Model",
    palette=palette,
    ax=axes[1],
    errorbar="sd",
    linewidth=2.2,
)
axes[1].set_title("Validation Loss Media (BCN20000 - 5 Fold)")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("Binary Cross-Entropy Loss")

plt.tight_layout()
p1 = "outputs/plots/bcn20000/bcn_loss_comparison.png"
plt.savefig(p1, dpi=300)
plt.close()
print(f"[OK] Salvato: {p1}")

# -------------------------------------------------------------------------
# GRAFICO 2: Evoluzione Epoca per Epoca di PR-AUC e MCC
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# PR-AUC
sns.lineplot(
    data=df_all,
    x="epoch",
    y="pr_auc",
    hue="Model",
    palette=palette,
    ax=axes[0],
    errorbar=None,
    linewidth=2.2,
    marker="o",
    markersize=4,
)
axes[0].set_title("Progressione PR-AUC (Precision-Recall AUC)")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("PR-AUC")

# MCC
sns.lineplot(
    data=df_all,
    x="epoch",
    y="mcc",
    hue="Model",
    palette=palette,
    ax=axes[1],
    errorbar=None,
    linewidth=2.2,
    marker="s",
    markersize=4,
)
axes[1].set_title("Progressione Matthews Correlation Coefficient (MCC)")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("MCC")

plt.tight_layout()
p2 = "outputs/plots/bcn20000/bcn_metrics_progression.png"
plt.savefig(p2, dpi=300)
plt.close()
print(f"[OK] Salvato: {p2}")

# -------------------------------------------------------------------------
# GRAFICO 3: Boxplot di Distribuzione delle Migliori Prestazioni sui Fold
# -------------------------------------------------------------------------
best_per_fold = (
    df_all.sort_values(by=["Model", "fold", "pr_auc"], ascending=False)
    .groupby(["Model", "fold"])
    .first()
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_to_plot = [("pr_auc", "PR-AUC"), ("recall", "Recall"), ("mcc", "MCC")]

for ax, (col, title) in zip(axes, metrics_to_plot):
    sns.boxplot(
        data=best_per_fold,
        x="Model",
        y=col,
        palette=palette,
        ax=ax,
        width=0.4,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
    )
    sns.stripplot(
        data=best_per_fold,
        x="Model",
        y=col,
        color="black",
        alpha=0.6,
        jitter=0.15,
        size=7,
        ax=ax,
    )
    ax.set_title(f"Distribuzione {title} sui Fold")
    ax.set_xlabel("")
    ax.set_ylabel(title)

plt.tight_layout()
p3 = "outputs/plots/bcn20000/bcn_boxplot_comparison.png"
plt.savefig(p3, dpi=300)
plt.close()
print(f"[OK] Salvato: {p3}")

# -------------------------------------------------------------------------
# GRAFICO 4: Sintesi a Barre delle Medie Globali
# -------------------------------------------------------------------------
summary = (
    best_per_fold.groupby("Model")[["pr_auc", "recall", "mcc"]]
    .mean()
    .reset_index()
)
summary_melted = pd.melt(
    summary,
    id_vars=["Model"],
    value_vars=["pr_auc", "recall", "mcc"],
    var_name="Metrica",
    value_name="Punteggio",
)
summary_melted["Metrica"] = summary_melted["Metrica"].replace(
    {"pr_auc": "PR-AUC", "recall": "Sensibilità (Recall)", "mcc": "MCC"}
)

plt.figure(figsize=(9, 5))
bar_plot = sns.barplot(
    data=summary_melted,
    x="Metrica",
    y="Punteggio",
    hue="Model",
    palette=palette,
)
plt.title("Confronto Complessivo BCN20000: CNN vs CNN+ViT (MobileViT-S)")
plt.ylabel("Punteggio Medio")
plt.ylim(0, 1.05)
plt.legend(title="Architettura", loc="upper right")

# Etichette numeriche sopra le barre
for p in bar_plot.patches:
    h = p.get_height()
    if h > 0:
        bar_plot.annotate(
            f"{h:.4f}",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

plt.tight_layout()
p4 = "outputs/plots/bcn20000/bcn_global_barchart.png"
plt.savefig(p4, dpi=300)
plt.close()
print(f"[OK] Salvato: {p4}")

print("\nGenerazione completata con successo nella cartella: outputs/plots/bcn20000/")