"""Stile grafico comune per le figure di addestramento della tesi (Cap. 4 e 5).

Colore = backbone: ResNet-50 azzurro, MobileViT-S arancione, VGG-16 grigio
(stessi colori delle figure SHAP del Cap. 6).
Stile = modalita' (Cap. 5): image-only linea tratteggiata / riempimento chiaro
rigato, multimodale linea continua / riempimento pieno.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#7f7f7f"
INK, MUTED, LIGHT = "#222222", "#666666", "#c8c8c8"

# Larghezza delle figure in pollici, pari circa a \textwidth: in LaTeX vanno
# incluse con width=\textwidth, cosi' il testo resta a 8-9 pt in tutte.
FULL_W = 6.3

METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "recall": "Sensibilità (τ = 0.5)",
    "mcc": "MCC",
}

MODEL_STYLE = {
    # Capitolo 4
    "VGG-16": {"color": GREY, "ls": "-", "hatch": None, "alpha": 0.85},
    "ResNet-50": {"color": BLUE, "ls": "-", "hatch": None, "alpha": 0.85},
    "MobileViT-S": {"color": ORANGE, "ls": "-", "hatch": None, "alpha": 0.85},
    # Capitolo 5
    "Caso 1 – ResNet-50": {"color": BLUE, "ls": "--", "hatch": "////", "alpha": 0.30},
    "Caso 2 – MobileViT-S": {"color": ORANGE, "ls": "--", "hatch": "////", "alpha": 0.30},
    "Caso 3 – ResNet-50 + clinici": {"color": BLUE, "ls": "-", "hatch": None, "alpha": 0.85},
    "Caso 4 – MobileViT-S + clinici": {"color": ORANGE, "ls": "-", "hatch": None, "alpha": 0.85},
}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "hatch.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------
def load_runs(files):
    """files: {nome_modello: percorso_csv} -> un unico DataFrame con colonna Model."""
    dfs = []
    for model, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV mancante: {path}")
        d = pd.read_csv(path)
        d["Model"] = model
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def best_per_fold(df, criterion):
    """Una riga per (modello, fold).

    max_pr_auc   -> epoca con PR-AUC massima (criterio delle tabelle del Cap. 4)
    min_val_loss -> epoca con validation loss minima, cioe' il checkpoint salvato
                    da train.py con l'early stopping (tabelle del Cap. 5)
    """
    df = df.sort_values(["Model", "fold", "epoch"]).reset_index(drop=True)
    if criterion == "max_pr_auc":
        idx = df.groupby(["Model", "fold"])["pr_auc"].idxmax()
    elif criterion == "min_val_loss":
        idx = df.groupby(["Model", "fold"])["val_loss"].idxmin()
    else:
        raise ValueError(f"criterio sconosciuto: {criterion}")
    return df.loc[idx].reset_index(drop=True)


def print_summary(best, models, cols):
    for m in models:
        b = best[best.Model == m]
        parts = [f"{c} {b[c].mean():.4f} ± {b[c].std():.4f}" for c in cols if c in b]
        print(f"  {m:<32} | " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Elementi grafici
# ---------------------------------------------------------------------------
def epoch_axis(ax, max_epoch):
    max_epoch = int(max_epoch)
    ax.set_xticks([1] + list(range(5, max_epoch + 1, 5)))
    ax.set_xlim(0.5, max_epoch + 0.5)
    ax.set_xlabel("Epoca")


def plot_curves(ax, df, col, models, band=True, band_alpha=0.15, min_folds=1):
    """Media sui fold per epoca, con banda +- deviazione standard.

    min_folds: mostra solo le epoche in cui almeno questo numero di fold e'
    ancora in addestramento (con l'early stopping le ultime epoche sono
    raggiunte da pochi fold e la loro media non e' rappresentativa).
    """
    xmax = 1
    for m in models:
        st = MODEL_STYLE[m]
        g = df[df.Model == m].groupby("epoch")[col].agg(["mean", "std", "count"])
        g = g[g["count"] >= min_folds]
        if g.empty:
            continue
        x = g.index.to_numpy()
        mu = g["mean"].to_numpy()
        sd = g["std"].fillna(0).to_numpy()
        ax.plot(x, mu, color=st["color"], ls=st["ls"], label=m)
        if band:
            ax.fill_between(x, mu - sd, mu + sd, color=st["color"], alpha=band_alpha, lw=0)
        xmax = max(xmax, x.max())
    epoch_axis(ax, xmax)


def box_panel(ax, best, col, models, tick_labels=None, tick_fontsize=8, seed=0):
    """Boxplot per fold: punti dei singoli fold e rombo bianco per la media."""
    rng = np.random.default_rng(seed)
    data = [best.loc[best.Model == m, col].to_numpy() for m in models]
    pos = np.arange(len(models))
    bp = ax.boxplot(data, positions=pos, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.2},
                    whiskerprops={"color": MUTED, "lw": 0.8},
                    capprops={"color": MUTED, "lw": 0.8})
    for patch, m in zip(bp["boxes"], models):
        st = MODEL_STYLE[m]
        patch.set(facecolor=to_rgba(st["color"], st["alpha"]), edgecolor=st["color"],
                  hatch=st["hatch"], lw=0.8)
    for k, vals in enumerate(data):
        ax.scatter(k + rng.uniform(-0.08, 0.08, len(vals)), vals, s=12, color=INK,
                   alpha=0.75, lw=0, zorder=3)
        ax.scatter(k, vals.mean(), marker="D", s=28, facecolor="white", edgecolor=INK,
                   lw=1.0, zorder=4)
    ax.set_xticks(pos, tick_labels or models, fontsize=tick_fontsize)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.grid(axis="x", visible=False)


def bar_panel(ax, stats, models, metrics):
    """stats[modello][metrica] = (media, std); std NaN -> nessuna barra d'errore."""
    x = np.arange(len(metrics))
    w = 0.8 / len(models)
    top = 0.0
    for k, m in enumerate(models):
        st = MODEL_STYLE[m]
        pos = x + (k - (len(models) - 1) / 2) * w
        means = np.array([stats[m][c][0] for c in metrics], dtype=float)
        stds = np.array([stats[m][c][1] for c in metrics], dtype=float)
        ax.bar(pos, means, width=w * 0.9, color=to_rgba(st["color"], st["alpha"]),
               edgecolor=st["color"], hatch=st["hatch"], lw=0.8, label=m)
        ok = ~np.isnan(stds)
        if ok.any():
            ax.errorbar(pos[ok], means[ok], yerr=stds[ok], fmt="none", ecolor=MUTED,
                        elinewidth=0.8, capsize=2)
        err = np.where(ok, stds, 0)
        for p, mu, e in zip(pos, means, err):
            ax.text(p, mu + e + 0.01, f"{mu:.3f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
        top = max(top, (means + err).max())
    ax.set_xticks(x, [METRIC_LABELS[c] for c in metrics])
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel("Media sui 5 fold")
    ax.grid(axis="x", visible=False)


def model_handles(models, kind="line"):
    hs = []
    for m in models:
        st = MODEL_STYLE[m]
        if kind == "line":
            hs.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=1.6, label=m))
        elif kind == "point":
            hs.append(Line2D([], [], marker="o", ls="none", markersize=5,
                             markerfacecolor=st["color"], markeredgecolor=INK,
                             markeredgewidth=0.4, label=m))
        else:
            hs.append(Patch(facecolor=to_rgba(st["color"], st["alpha"]), edgecolor=st["color"],
                            hatch=st["hatch"], lw=0.8, label=m))
    return hs


def mean_handle(label="Media sui 5 fold"):
    return Line2D([], [], marker="D", ls="none", markersize=5, markerfacecolor="white",
                  markeredgecolor=INK, markeredgewidth=1.0, label=label)


def top_legend(fig, handles, ncol):
    """Legenda comune sopra la figura (chiamare dopo tight_layout)."""
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol, frameon=False)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {path}")
