import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Stile tipografico per tesi magistrale
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

# 1. Caricamento file metriche
files = {
    "ResNet-50 (CNN)": "outputs/metrics_bcn_cnn.csv",
    "MobileViT-S (Hybrid)": "outputs/metrics_bcn_hybrid.csv",
}

# Blu per CNN, Arancione per Hybrid
palette = {
    "ResNet-50 (CNN)": "#1f77b4",
    "MobileViT-S (Hybrid)": "#ff7f0e",
}
model_order = ["ResNet-50 (CNN)", "MobileViT-S (Hybrid)"]

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
      "Nessun file CSV trovato. Verifica il percorso in outputs/"
  )

df_all = pd.concat(dfs, ignore_index=True)
out_dir = "figure"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------------------
# GRAFICO 1: Curve di Loss (Training vs Validation)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

sns.lineplot(
    data=df_all,
    x="epoch",
    y="train_loss",
    hue="Model",
    hue_order=model_order,
    palette=palette,
    ax=axes[0],
    errorbar="sd",
    linewidth=2.2,
)
axes[0].set_title("Training Loss Media (BCN20000 - 5 Fold)")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("Binary Cross-Entropy Loss")

sns.lineplot(
    data=df_all,
    x="epoch",
    y="val_loss",
    hue="Model",
    hue_order=model_order,
    palette=palette,
    ax=axes[1],
    errorbar="sd",
    linewidth=2.2,
)
axes[1].set_title("Validation Loss Media (BCN20000 - 5 Fold)")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("Binary Cross-Entropy Loss")

plt.tight_layout()
p1 = os.path.join(out_dir, "bcn_loss_comparison.png")
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
    ("mcc", "Matthews Corr. Coeff. (MCC)"),
]

for idx, (col, title) in enumerate(metrics_prog):
  sns.lineplot(
      data=df_all,
      x="epoch",
      y=col,
      hue="Model",
      hue_order=model_order,
      palette=palette,
      ax=axes[idx],
      errorbar=None,
      linewidth=2.2,
  )
  axes[idx].set_title(title, fontweight="bold")
  axes[idx].set_xlabel("Epoca")
  axes[idx].set_ylabel(title)

plt.tight_layout()
p2 = os.path.join(out_dir, "bcn_metrics_progression.png")
plt.savefig(p2, dpi=300)
plt.close()
print(f"[OK] Salvato: {p2}")

# -------------------------------------------------------------------------
# GRAFICO 3: Boxplot sui 5 Fold (ResNet-50 Blu a Sx, MobileViT Arancione a Dx)
# -------------------------------------------------------------------------
best_per_fold = (
    df_all.sort_values(by=["Model", "fold", "pr_auc"], ascending=False)
    .groupby(["Model", "fold"])
    .first()
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = [
    ("pr_auc", "PR-AUC"),
    ("recall", "Recall (Sensibilità)"),
    ("mcc", "Matthews Corr. Coeff. (MCC)"),
]

for ax, (col, title) in zip(axes, metrics_to_plot):
  sns.boxplot(
      data=best_per_fold,
      x="Model",
      y=col,
      order=model_order,
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
      order=model_order,
      color="black",
      alpha=0.6,
      jitter=0.15,
      size=7,
      ax=ax,
  )
  ax.set_title(f"Distribuzione {title}", fontweight="bold")
  ax.set_xlabel("")
  ax.set_ylabel(title)

plt.suptitle(
    "Variabilità e Robustezza sui 5 Fold (Migliori Pesi)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
p3 = os.path.join(out_dir, "bcn_boxplot_comparison.png")
plt.savefig(p3, dpi=300)
plt.close()
print(f"[OK] Salvato: {p3}")

# -------------------------------------------------------------------------
# GRAFICO 4: Sintesi a Barre Globali con Barre di Errore
# -------------------------------------------------------------------------
summary = (
    best_per_fold.groupby("Model")[["pr_auc", "recall", "mcc"]]
    .agg(["mean", "std"])
    .reindex(model_order)
)

labels = ["PR-AUC", "Recall", "MCC"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(9, 5.5))
cnn_means = [
    summary.loc["ResNet-50 (CNN)", (m, "mean")]
    for m in ["pr_auc", "recall", "mcc"]
]
cnn_stds = [
    summary.loc["ResNet-50 (CNN)", (m, "std")]
    for m in ["pr_auc", "recall", "mcc"]
]
hyb_means = [
    summary.loc["MobileViT-S (Hybrid)", (m, "mean")]
    for m in ["pr_auc", "recall", "mcc"]
]
hyb_stds = [
    summary.loc["MobileViT-S (Hybrid)", (m, "std")]
    for m in ["pr_auc", "recall", "mcc"]
]

r1 = plt.bar(
    x - width / 2,
    cnn_means,
    width,
    yerr=cnn_stds,
    capsize=5,
    label="ResNet-50 (CNN)",
    color="#1f77b4",
    alpha=0.9,
)
r2 = plt.bar(
    x + width / 2,
    hyb_means,
    width,
    yerr=hyb_stds,
    capsize=5,
    label="MobileViT-S (Hybrid)",
    color="#ff7f0e",
    alpha=0.9,
)

plt.ylabel("Punteggio Medio")
plt.title(
    "Benchmark Finale BCN20000: ResNet-50 vs MobileViT-S (5 Fold CV)",
    fontweight="bold",
)
plt.xticks(x, labels, fontweight="bold")
plt.ylim(0, 1.05)
plt.legend(frameon=True)

for rect in list(r1) + list(r2):
  h = rect.get_height()
  plt.text(
      rect.get_x() + rect.get_width() / 2.0,
      h + 0.02,
      f"{h:.3f}",
      ha="center",
      va="bottom",
      fontsize=9,
      fontweight="bold",
  )

plt.tight_layout()
p4 = os.path.join(out_dir, "bcn_global_barchart.png")
plt.savefig(p4, dpi=300)
plt.close()
print(f"[OK] Salvato: {p4}")

print(f"\n[SUCCESSO] Tutti i 4 plot salvati correttamente in {out_dir}/")
