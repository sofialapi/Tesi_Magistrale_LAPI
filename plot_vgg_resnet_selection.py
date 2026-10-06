import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CSV per-epoca (fold, epoch, train_loss, val_loss, pr_auc, recall, mcc)
# ResNet-50 / MobileViT-S = run del Capitolo 4
# Se il diff mostra che VGG su BCN e' stato addestrato su immagini raw,
# sostituisci BCN_RESNET con "outputs/raw/metrics_BCN_raw_cnn.csv"
# =====================================================================
BCN_RESNET = "outputs/metrics_bcn_cnn.csv"

RUNS = {
    "BCN20000": {
        "VGG-16":      "outputs/vgg/metrics/metrics_vgg_bcn20000_cnn.csv",
        "ResNet-50":   BCN_RESNET,
        "MobileViT-S": "outputs/metrics_bcn_hybrid.csv",
    },
    "HAM10000": {
        "VGG-16":      "outputs/vgg/metrics/metrics_vgg_ham10000_cnn.csv",
        "ResNet-50":   "outputs/metrics_ham10000_cnn.csv",
        "MobileViT-S": "outputs/metrics_ham10000_hybrid.csv",
    },
}
# Secondo addestramento di MobileViT-S (stesso protocollo): solo controllo di riproducibilita'
CONTROL = {
    "BCN20000": "outputs/vgg/metrics/metrics_vgg_bcn20000_hybrid.csv",
    "HAM10000": "outputs/vgg/metrics/metrics_vgg_ham10000_hybrid.csv",
}
# Riferimento stampato a terminale (non usato nei plot)
REFERENCE = {"ResNet-50 BCN raw (Tab. 4.3)": "outputs/raw/metrics_BCN_raw_cnn.csv"}

ORDER = ["VGG-16", "ResNet-50", "MobileViT-S"]
PALETTE = {"VGG-16": "#7f7f7f", "ResNet-50": "#1f77b4", "MobileViT-S": "#ff7f0e"}
METRICS = [("pr_auc", "PR-AUC"), ("recall", "Recall (τ = 0.5)"), ("mcc", "MCC")]
OUT_DIR = "figure/vgg"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"]})
sns.set_theme(style="whitegrid")


def load(path, model, dataset):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV mancante: {path}")
    df = pd.read_csv(path)
    df["Model"] = model
    df["Dataset"] = dataset
    return df


def best_per_fold(df):
    # epoca con PR-AUC massima per fold (stesso criterio di plot_vgg_comparison.py)
    return df.loc[df.groupby("fold")["pr_auc"].idxmax()].reset_index(drop=True)


def fmt(s):
    return f"{s.mean():.4f} ± {s.std():.4f}"


hist_list, best_list = [], []
for ds, runs in RUNS.items():
    for m, p in runs.items():
        h = load(p, m, ds)
        hist_list.append(h)
        best_list.append(best_per_fold(h))
hist = pd.concat(hist_list, ignore_index=True)
best = pd.concat(best_list, ignore_index=True)

# ---------------------------------------------------------------------
# 1. Riepilogo numerico, confronto appaiato, controllo MobileViT
# ---------------------------------------------------------------------
print("\n=== Media ± dev. std sui 5 fold (epoca con PR-AUC massima per fold) ===")
for ds in RUNS:
    print(f"\n[{ds}]")
    for m in ORDER:
        b = best[(best.Dataset == ds) & (best.Model == m)]
        print(f"{m:<12} | PR-AUC {fmt(b.pr_auc)} | Recall {fmt(b.recall)} | MCC {fmt(b.mcc)}")

    v = best[(best.Dataset == ds) & (best.Model == "VGG-16")].set_index("fold")
    r = best[(best.Dataset == ds) & (best.Model == "ResNet-50")].set_index("fold")
    if set(v.index) != set(r.index):
        print("  [ATTENZIONE] indici di fold diversi: confronto appaiato non valido")
    else:
        for col, _ in METRICS:
            d = r[col] - v.loc[r.index, col]
            print(f"  ResNet-50 − VGG-16 {col:<7}: Δ medio {d.mean():+.4f} | ResNet migliore in {(d > 0).sum()}/{len(d)} fold")
        print("  Intervalli per fold PR-AUC: "
              f"VGG [{v.pr_auc.min():.3f}, {v.pr_auc.max():.3f}]  ResNet [{r.pr_auc.min():.3f}, {r.pr_auc.max():.3f}]")

    c = best_per_fold(load(CONTROL[ds], "ctrl", ds))
    mv = best[(best.Dataset == ds) & (best.Model == "MobileViT-S")]
    print(f"  Controllo MobileViT-S (2° addestramento): PR-AUC {fmt(c.pr_auc)} | Recall {fmt(c.recall)} | MCC {fmt(c.mcc)}")
    print(f"    scarto PR-AUC rispetto al Cap. 4: {c.pr_auc.mean() - mv.pr_auc.mean():+.4f}")

for name, p in REFERENCE.items():
    if os.path.exists(p):
        b = best_per_fold(pd.read_csv(p))
        print(f"\n[Riferimento] {name}: PR-AUC {fmt(b.pr_auc)} | Recall {fmt(b.recall)} | MCC {fmt(b.mcc)}")

print("\n=== Righe LaTeX (Dataset & Modello & PR-AUC & Recall & MCC) ===")
for ds in RUNS:
    for m in ORDER:
        b = best[(best.Dataset == ds) & (best.Model == m)]
        cells = " & ".join(f"{b[c].mean():.3f} $\\pm$ {b[c].std():.3f}" for c, _ in METRICS)
        print(f"{ds} & {m} & {cells} \\\\")

# ---------------------------------------------------------------------
# 2. Figura principale: valori per fold, linee che uniscono lo stesso fold
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for i, ds in enumerate(RUNS):
    sub = best[best.Dataset == ds]
    for j, (col, title) in enumerate(METRICS):
        ax = axes[i, j]
        for _, g in sub.groupby("fold"):
            g = g.set_index("Model").reindex(ORDER)
            ax.plot(range(len(ORDER)), g[col].values, color="0.75", lw=1, zorder=1)
        for k, m in enumerate(ORDER):
            vals = sub[sub.Model == m][col].values
            ax.scatter(np.full(len(vals), k), vals, color=PALETTE[m], s=50,
                       edgecolor="black", linewidth=0.4, alpha=0.9, zorder=2)
            ax.scatter(k, vals.mean(), marker="D", s=75, color="white",
                       edgecolor="black", linewidth=1.2, zorder=3)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER)
        ax.set_xlim(-0.4, len(ORDER) - 0.6)
        ax.set_title(f"{title} – {ds}", fontweight="bold")
        ax.set_ylabel(title)
plt.suptitle("Confronto per fold: VGG-16, ResNet-50 e MobileViT-S (◇ = media)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "selezione_backbone_per_fold.png")
plt.savefig(p1, dpi=300)
plt.close()
print(f"\n[OK] Salvato: {p1}")

# ---------------------------------------------------------------------
# 3. Figura opzionale: loss VGG-16 vs ResNet-50
# ---------------------------------------------------------------------
cnn = hist[hist.Model.isin(["VGG-16", "ResNet-50"])]
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for i, ds in enumerate(RUNS):
    for j, (col, lab) in enumerate([("train_loss", "Training loss"), ("val_loss", "Validation loss")]):
        ax = axes[i, j]
        sns.lineplot(data=cnn[cnn.Dataset == ds], x="epoch", y=col, hue="Model",
                     hue_order=["VGG-16", "ResNet-50"], palette=PALETTE,
                     errorbar="sd", linewidth=2.2, ax=ax)
        ax.set_title(f"{lab} – {ds} (media ± std sui 5 fold)", fontweight="bold")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("BCE pesata")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.legend(title=None)
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "selezione_backbone_loss.png")
plt.savefig(p2, dpi=300)
plt.close()
print(f"[OK] Salvato: {p2}")
