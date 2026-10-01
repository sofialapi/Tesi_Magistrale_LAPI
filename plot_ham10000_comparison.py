import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})


def main():
  cnn_csv = "outputs/metrics_ham10000_cnn.csv"
  hyb_csv = "outputs/metrics_ham10000_hybrid.csv"
  out_dir = "figure/ham10000"
  os.makedirs(out_dir, exist_ok=True)

  if not os.path.exists(cnn_csv) or not os.path.exists(hyb_csv):
    raise FileNotFoundError(
        "File metriche mancanti in outputs/: assicurati che esistano"
        f" {cnn_csv} e {hyb_csv}"
    )

  df_cnn = pd.read_csv(cnn_csv)
  df_hyb = pd.read_csv(hyb_csv)

  df_cnn["Model"] = "ResNet-50 (CNN)"
  df_hyb["Model"] = "MobileViT-S (Hybrid)"

  # 1. Plot Confronto Curve di Loss (Train vs Val)
  plt.figure(figsize=(10, 5))
  cnn_mean_loss = (
      df_cnn.groupby("epoch")[["train_loss", "val_loss"]].mean().reset_index()
  )
  hyb_mean_loss = (
      df_hyb.groupby("epoch")[["train_loss", "val_loss"]].mean().reset_index()
  )

  plt.plot(
      cnn_mean_loss["epoch"],
      cnn_mean_loss["train_loss"],
      "--",
      color="#1f77b4",
      label="ResNet-50 Train Loss",
  )
  plt.plot(
      cnn_mean_loss["epoch"],
      cnn_mean_loss["val_loss"],
      "-",
      color="#1f77b4",
      linewidth=2,
      label="ResNet-50 Val Loss",
  )
  plt.plot(
      hyb_mean_loss["epoch"],
      hyb_mean_loss["train_loss"],
      "--",
      color="#ff7f0e",
      label="MobileViT-S Train Loss",
  )
  plt.plot(
      hyb_mean_loss["epoch"],
      hyb_mean_loss["val_loss"],
      "-",
      color="#ff7f0e",
      linewidth=2,
      label="MobileViT-S Val Loss",
  )

  plt.title(
      "Andamento Curve di Loss su HAM10000 (Media 5 Fold)",
      fontsize=13,
      fontweight="bold",
  )
  plt.xlabel("Epoca", fontsize=11)
  plt.ylabel("BCE Loss (Ponderata)", fontsize=11)
  plt.legend(frameon=True)
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "ham_loss_comparison.png"), dpi=300)
  plt.close()
  print(f"--> Salvato: {out_dir}/ham_loss_comparison.png")

  # 2. Plot Progressione Metriche Epocali (PR-AUC, Recall, MCC)
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  metrics = [
      ("pr_auc", "PR-AUC"),
      ("recall", "Recall (Sensibilità)"),
      ("mcc", "Matthews Corr. Coeff. (MCC)"),
  ]

  for idx, (m_col, m_title) in enumerate(metrics):
    m_cnn = df_cnn.groupby("epoch")[m_col].agg(["mean", "std"]).reset_index()
    m_hyb = df_hyb.groupby("epoch")[m_col].agg(["mean", "std"]).reset_index()

    axes[idx].plot(
        m_cnn["epoch"], m_cnn["mean"], label="ResNet-50", color="#1f77b4", lw=2
    )
    axes[idx].fill_between(
        m_cnn["epoch"],
        m_cnn["mean"] - m_cnn["std"],
        m_cnn["mean"] + m_cnn["std"],
        alpha=0.15,
        color="#1f77b4",
    )

    axes[idx].plot(
        m_hyb["epoch"],
        m_hyb["mean"],
        label="MobileViT-S",
        color="#ff7f0e",
        lw=2,
    )
    axes[idx].fill_between(
        m_hyb["epoch"],
        m_hyb["mean"] - m_hyb["std"],
        m_hyb["mean"] + m_hyb["std"],
        alpha=0.15,
        color="#ff7f0e",
    )

    axes[idx].set_title(m_title, fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Epoca")
    axes[idx].legend(loc="lower right")

  plt.suptitle(
      "Confronto Dinamico Prestazioni su HAM10000 lungo le 20 Epoche",
      fontsize=14,
      fontweight="bold",
  )
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "ham_metrics_progression.png"), dpi=300)
  plt.close()
  print(f"--> Salvato: {out_dir}/ham_metrics_progression.png")

  # 3. Boxplot Comparativo sui Fold (Miglior epoca per fold)
  df_all = pd.concat([df_cnn, df_hyb], ignore_index=True)
  best_per_fold = df_all.loc[
      df_all.groupby(["Model", "fold"])["pr_auc"].idxmax()
  ]

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  for idx, (m_col, m_title) in enumerate(metrics):
    sns.boxplot(
        ax=axes[idx],
        x="Model",
        y=m_col,
        data=best_per_fold,
        palette=["#1f77b4", "#ff7f0e"],
        width=0.4,
    )
    sns.stripplot(
        ax=axes[idx],
        x="Model",
        y=m_col,
        data=best_per_fold,
        color="black",
        alpha=0.6,
        jitter=0.2,
        size=7,
    )
    axes[idx].set_title(
        f"Distribuzione {m_title}", fontsize=12, fontweight="bold"
    )
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel(m_title)

  plt.suptitle(
      "Variabilità e Robustezza sui 5 Fold - HAM10000",
      fontsize=14,
      fontweight="bold",
  )
  plt.tight_layout()
  plt.savefig(os.path.join(out_dir, "ham_boxplot_comparison.png"), dpi=300)
  plt.close()
  print(f"--> Salvato: {out_dir}/ham_boxplot_comparison.png")

  # 4. Barchart Riassuntivo Globale
  summary = best_per_fold.groupby("Model")[["pr_auc", "recall", "mcc"]].agg(
      ["mean", "std"]
  )
  labels = ["PR-AUC", "Recall", "MCC"]
  cnn_means = [
      summary.loc["ResNet-50 (CNN)", (col, "mean")]
      for col in ["pr_auc", "recall", "mcc"]
  ]
  cnn_stds = [
      summary.loc["ResNet-50 (CNN)", (col, "std")]
      for col in ["pr_auc", "recall", "mcc"]
  ]
  hyb_means = [
      summary.loc["MobileViT-S (Hybrid)", (col, "mean")]
      for col in ["pr_auc", "recall", "mcc"]
  ]
  hyb_stds = [
      summary.loc["MobileViT-S (Hybrid)", (col, "std")]
      for col in ["pr_auc", "recall", "mcc"]
  ]

  x = np.arange(len(labels))
  width = 0.35

  plt.figure(figsize=(9, 5.5))
  rects1 = plt.bar(
      x - width / 2,
      cnn_means,
      width,
      yerr=cnn_stds,
      capsize=5,
      label="ResNet-50 (CNN)",
      color="#1f77b4",
      alpha=0.9,
  )
  rects2 = plt.bar(
      x + width / 2,
      hyb_means,
      width,
      yerr=hyb_stds,
      capsize=5,
      label="MobileViT-S (Hybrid)",
      color="#ff7f0e",
      alpha=0.9,
  )

  plt.ylabel("Punteggio Medio", fontsize=11)
  plt.title(
      "Benchmark Finale HAM10000: ResNet-50 vs MobileViT-S (5 Fold CV)",
      fontsize=13,
      fontweight="bold",
  )
  plt.xticks(x, labels, fontsize=11, fontweight="bold")
  plt.ylim(0, 1.05)
  plt.legend(frameon=True)

  for rect in rects1 + rects2:
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
  plt.savefig(os.path.join(out_dir, "ham_global_barchart.png"), dpi=300)
  plt.close()
  print(f"--> Salvato: {out_dir}/ham_global_barchart.png")


if __name__ == "__main__":
  main()
