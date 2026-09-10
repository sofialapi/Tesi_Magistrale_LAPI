import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configurazione stile grafico conforme a pubblicazione scientifica / tesi
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})
sns.set_theme(style="whitegrid")

# 1. Caricamento e strutturazione dei dati
files = {
    "CNN Only": "outputs/metrics_cnn_only.csv",
    "Hybrid Only": "outputs/metrics_hybrid_only.csv",
    "CNN Multimodal": "outputs/metrics_multimodal.csv",
    "Hybrid Multimodal": "outputs/metrics_hybrid_multimodal.csv",
}

palette = {
    "CNN Only": "#4C72B0",
    "Hybrid Only": "#55A868",
    "CNN Multimodal": "#DD8452",
    "Hybrid Multimodal": "#8172B3",
}

dataframes = []
for name, path in files.items():
    if os.path.exists(path):
        df_temp = pd.read_csv(path)
        df_temp["Model"] = name
        dataframes.append(df_temp)
    else:
        print(f"[ATTENZIONE] File non trovato: {path}")

if not dataframes:
    raise FileNotFoundError(
        "Nessun file CSV trovato. Assicurati di aver fatto 'git pull origin main'."
    )

df_all = pd.concat(dataframes, ignore_index=True)
os.makedirs("outputs/plots", exist_ok=True)

# -------------------------------------------------------------
# GRAFICO 1: Curve Medie di Training Loss e Validation Loss
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

# Training Loss
sns.lineplot(
    data=df_all,
    x="epoch",
    y="train_loss",
    hue="Model",
    palette=palette,
    ax=axes[0],
    errorbar=None,
    linewidth=2,
)
axes[0].set_title("Training Loss Media (5-Fold CV)")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("Focal Loss")

# Validation Loss
sns.lineplot(
    data=df_all,
    x="epoch",
    y="val_loss",
    hue="Model",
    palette=palette,
    ax=axes[1],
    errorbar=None,
    linewidth=2,
)
axes[1].set_title("Validation Loss Media (5-Fold CV)")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("Focal Loss")

plt.tight_layout()
plot1_path = "outputs/plots/loss_curves_comparison.png"
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"[OK] Salvato: {plot1_path}")

# -------------------------------------------------------------
# GRAFICO 2: Curve di Apprendimento di PR-AUC e Recall
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

# PR-AUC
sns.lineplot(
    data=df_all,
    x="epoch",
    y="pr_auc",
    hue="Model",
    palette=palette,
    ax=axes[0],
    errorbar=None,
    linewidth=2,
)
axes[0].set_title("Progressione Media PR-AUC")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("PR-AUC")

# Recall
sns.lineplot(
    data=df_all,
    x="epoch",
    y="recall",
    hue="Model",
    palette=palette,
    ax=axes[1],
    errorbar=None,
    linewidth=2,
)
axes[1].set_title("Progressione Media Sensibilità (Recall)")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("Recall")

plt.tight_layout()
plot2_path = "outputs/plots/metrics_progression_comparison.png"
plt.savefig(plot2_path, dpi=300)
plt.close()
print(f"[OK] Salvato: {plot2_path}")

# -------------------------------------------------------------
# GRAFICO 3: Boxplot delle Prestazioni sui 5 Fold (Migliori Valori)
# -------------------------------------------------------------
# Estraiamo la miglior epoca per ciascun fold (criterio: massima PR-AUC)
best_per_fold = (
    df_all.sort_values(by=["Model", "fold", "pr_auc"], ascending=False)
    .groupby(["Model", "fold"])
    .first()
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot PR-AUC
sns.boxplot(
    data=best_per_fold,
    x="Model",
    y="pr_auc",
    palette=palette,
    ax=axes[0],
    width=0.45,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
    },
)
sns.stripplot(
    data=best_per_fold,
    x="Model",
    y="pr_auc",
    color="black",
    alpha=0.6,
    jitter=0.2,
    size=6,
    ax=axes[0],
)
axes[0].set_title("Distribuzione PR-AUC sui 5 Fold")
axes[0].set_xlabel("")
axes[0].set_ylabel("PR-AUC")
axes[0].tick_params(axis="x", rotation=15)

# Boxplot Recall
sns.boxplot(
    data=best_per_fold,
    x="Model",
    y="recall",
    palette=palette,
    ax=axes[1],
    width=0.45,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
    },
)
sns.stripplot(
    data=best_per_fold,
    x="Model",
    y="recall",
    color="black",
    alpha=0.6,
    jitter=0.2,
    size=6,
    ax=axes[1],
)
axes[1].set_title("Distribuzione Sensibilità (Recall) sui 5 Fold")
axes[1].set_xlabel("")
axes[1].set_ylabel("Recall")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plot3_path = "outputs/plots/boxplot_cross_validation.png"
plt.savefig(plot3_path, dpi=300)
plt.close()
print(f"[OK] Salvato: {plot3_path}")

# -------------------------------------------------------------
# GRAFICO 4: Sintesi Finale a Barre delle Medie Globali (Inclusi MCC)
# -------------------------------------------------------------
# Riepilogo finale aggregato dei log
summary_data = {
    "Modello": [
        "CNN Only",
        "Hybrid Only",
        "CNN Multimodal",
        "Hybrid Multimodal",
    ],
    "PR-AUC": [0.4208, 0.5062, 0.4282, 0.4869],
    "Sensibilità": [0.0380, 0.1348, 0.4174, 0.4224],
    "MCC": [0.1545, 0.3350, 0.4381, 0.4677],
}
df_summary = pd.DataFrame(summary_data)
df_summary_melted = pd.melt(
    df_summary,
    id_vars=["Modello"],
    value_vars=["PR-AUC", "Sensibilità", "MCC"],
    var_name="Metrica",
    value_name="Punteggio",
)

plt.figure(figsize=(10, 5))
bar_plot = sns.barplot(
    data=df_summary_melted,
    x="Metrica",
    y="Punteggio",
    hue="Modello",
    palette=palette,
)
plt.title("Confronto Complessivo delle Metriche Chiave sui 5 Fold")
plt.ylabel("Valore Score [0 - 1]")
plt.ylim(0, 0.6)
plt.legend(title="Architettura", loc="upper left")

# Aggiunta etichette numeriche sopra le barre
for p in bar_plot.patches:
    height = p.get_height()
    if height > 0:
        bar_plot.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
            xytext=(0, 3),
            textcoords="offset points",
        )

plt.tight_layout()
plot4_path = "outputs/plots/global_metrics_barchart.png"
plt.savefig(plot4_path, dpi=300)
plt.close()
print(f"[OK] Salvato: {plot4_path}")

print(
    "\nTutti i grafici ad alta risoluzione (300 DPI) sono stati generati nella cartella outputs/plots/"
)