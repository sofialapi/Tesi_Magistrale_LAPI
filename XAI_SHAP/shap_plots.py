"""Figure e tabelle riassuntive dei valori SHAP (legge i CSV di shap_run.py).

Uso: python -m XAI_SHAP.shap_plots
Non richiede GPU né il modello.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from XAI_SHAP import shap_config as cfg  # noqa: E402

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
})
INK, MUTED = "#222222", "#666666"
LOGIT = "contributo al log-odds di malignità"


def load(case):
    path = os.path.join(cfg.OUT_DIR, f"shap_values_{case}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def save(fig, name):
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    out = os.path.join(cfg.FIG_DIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  salvata: {os.path.relpath(out, cfg.ROOT)}")


def importance_table(data):
    """mean|phi| per variabile: media sui 5 fold e deviazione standard tra fold."""
    rows = []
    for case, d in data.items():
        per_fold = d.groupby("fold")[[f"phi_{v}" for v in cfg.VAR_ORDER]].apply(lambda x: x.abs().mean())
        for v in cfg.VAR_ORDER:
            col = per_fold[f"phi_{v}"]
            rows.append({"case": case, "variable": v, "mean_abs_phi": d[f"phi_{v}"].abs().mean(),
                         "std_between_folds": col.std(ddof=1)})
    return pd.DataFrame(rows)


def fig_importance(data, imp):
    cases = list(data)
    order = (imp[imp.case == cases[0]].sort_values("mean_abs_phi")["variable"].tolist())
    y = np.arange(len(order))
    h = 0.8 / len(cases)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(order) + 1.2))
    for k, case in enumerate(cases):
        sub = imp[imp.case == case].set_index("variable").loc[order]
        pos = y + (k - (len(cases) - 1) / 2) * h
        ax.barh(pos, sub["mean_abs_phi"], height=h * 0.9, color=cfg.CASE_COLORS[case],
                xerr=sub["std_between_folds"], error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2},
                label=cfg.CASES[case]["label"])
        for p, val, err in zip(pos, sub["mean_abs_phi"], sub["std_between_folds"]):
            ax.text(val + err + 0.01 * ax.get_xlim()[1] + 0.005, p, f"{val:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [cfg.VAR_LABELS[v] for v in order])
    ax.set_xlabel("media di |SHAP| sulle lesioni di validazione (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    save(fig, "shap_importanza_variabili.png")


def _color_values(d, var):
    """Valori normalizzati in [0, 1] per colorare i punti (NaN per le categoriali)."""
    if var in cfg.CAT_VARS:
        return np.full(len(d), np.nan)
    x = pd.to_numeric(d[f"val_{var}"], errors="coerce")
    if var == "tbp_lv_areaMM2":
        x = np.log10(x)
    lo, hi = np.nanpercentile(x, [5, 95])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).to_numpy()


def fig_beeswarm(data, imp):
    cases = list(data)
    order = imp[imp.case == cases[0]].sort_values("mean_abs_phi")["variable"].tolist()
    cmap = plt.get_cmap("coolwarm")
    rng = np.random.default_rng(0)
    xmax = max(np.nanpercentile(np.abs(d[[f"phi_{v}" for v in order]].values), 99.8) for d in data.values())
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 0.55 * len(order) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        d = data[case]
        for i, v in enumerate(order):
            phi = d[f"phi_{v}"].to_numpy()
            c = _color_values(d, v)
            jitter = rng.uniform(-0.3, 0.3, len(phi)) * np.minimum(1, 4 * np.abs(phi) / (xmax + 1e-12) + 0.25)
            nan = np.isnan(c)
            ax.scatter(phi[nan], i + jitter[nan], s=3, color="#a6a6a6", alpha=0.5, lw=0, rasterized=True)
            ax.scatter(phi[~nan], i + jitter[~nan], s=3, c=c[~nan], cmap=cmap, vmin=0, vmax=1,
                       alpha=0.7, lw=0, rasterized=True)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlim(-xmax * 1.05, xmax * 1.05)
        ax.set_yticks(range(len(order)), [cfg.VAR_LABELS[v] for v in order])
        ax.grid(axis="y", visible=False)
        ax.set_title(cfg.CASES[case]["label"])
        ax.set_xlabel(f"valore SHAP ({LOGIT})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.03, ticks=[0, 1])
    cb.ax.set_yticklabels(["basso", "alto"])
    cb.set_label("valore della variabile numerica\n(grigio: variabili categoriali)", color=MUTED)
    save(fig, "shap_beeswarm.png")


def fig_dependence(data):
    nums = cfg.NUM_VARS
    cases = list(data)
    fig, axes = plt.subplots(len(cases), len(nums), figsize=(3.2 * len(nums), 2.8 * len(cases)), squeeze=False)
    for r, case in enumerate(cases):
        d = data[case]
        for c, v in enumerate(nums):
            ax = axes[r, c]
            x = pd.to_numeric(d[f"val_{v}"], errors="coerce").to_numpy()
            phi = d[f"phi_{v}"].to_numpy()
            for cls in (0, 1):
                m = (d["target"].to_numpy() == cls) & ~np.isnan(x)
                ax.scatter(x[m], phi[m], s=4 if cls == 0 else 9, color=cfg.CLASS_COLORS[cls],
                           alpha=0.35 if cls == 0 else 0.8, lw=0, rasterized=True,
                           label="benigna" if cls == 0 else "maligna", zorder=2 + cls)
            ok = ~np.isnan(x)
            edges = np.unique(np.nanpercentile(x[ok], np.linspace(0, 100, 16)))
            if len(edges) > 2:
                idx = np.clip(np.digitize(x[ok], edges[1:-1]), 0, len(edges) - 2)
                ks = [k for k in range(len(edges) - 1) if np.any(idx == k)]
                med = [np.median(phi[ok][idx == k]) for k in ks]
                mid = [np.median(x[ok][idx == k]) for k in ks]
                ax.plot(mid, med, color=INK, lw=1.6, zorder=5, label="mediana per intervallo")
            if v == "tbp_lv_areaMM2":
                ax.set_xscale("log")
            ax.axhline(0, color=MUTED, lw=0.7)
            ax.set_xlabel(cfg.VAR_LABELS[v])
            if c == 0:
                ax.set_ylabel(f"SHAP\n{cfg.CASES[case]['short']}")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best", markerscale=2)
    fig.tight_layout()
    save(fig, "shap_dipendenza_numeriche.png")


def fig_categorical(data):
    """Boxplot SHAP per categoria; sotto ogni categoria, numerosità e tasso empirico di malignità."""
    cases = list(data)
    missing = "n.d."
    fig, axes = plt.subplots(1, len(cfg.CAT_VARS), figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 2.3]})
    w = 0.8 / len(cases)
    ref = data[cases[0]]
    for ax, var in zip(axes, cfg.CAT_VARS):
        lab_ref = ref[f"val_{var}"].fillna(missing)
        cats = [c for c in lab_ref.value_counts().index if c != missing] + ([missing] if (lab_ref == missing).any() else [])
        x = np.arange(len(cats))
        for k, case in enumerate(cases):
            d = data[case]
            lab = d[f"val_{var}"].fillna(missing)
            groups = [d.loc[lab == cat, f"phi_{var}"].to_numpy() for cat in cats]
            pos = x + (k - (len(cases) - 1) / 2) * w
            bp = ax.boxplot(groups, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                            medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": MUTED},
                            capprops={"color": MUTED})
            for patch in bp["boxes"]:
                patch.set(facecolor=cfg.CASE_COLORS[case], alpha=0.75, edgecolor=MUTED)
            ax.plot([], [], "s", color=cfg.CASE_COLORS[case], label=cfg.CASES[case]["short"])
        ticks = []
        for c in cats:
            m = lab_ref == c
            name = "n.d. (imputato)" if c == missing else cfg.CAT_LABELS.get(c, c)
            ticks.append(f"{name}\nn={m.sum()}\n{ref.loc[m, 'target'].mean():.1%} mal.")
        ax.set_xticks(x, ticks, fontsize=7.5)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_title(cfg.VAR_LABELS[var])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("valore SHAP\n(contributo al log-odds di malignità)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "shap_categoriali.png")


def fig_modality(data, folds):
    cases = list(data)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    # Sinistra: |phi| medio di immagine e clinica, separato per classe.
    labels = ["benigne", "maligne"]
    x = np.arange(len(labels))
    w = 0.8 / (2 * len(cases))
    k = 0
    for case in cases:
        d = data[case]
        for comp, hatch in (("phi_img", None), ("phi_tab", "///")):
            vals = [d.loc[d.target == cls, comp].abs().mean() for cls in (0, 1)]
            pos = x + (k - (2 * len(cases) - 1) / 2) * w
            a1.bar(pos, vals, width=w * 0.9, color=cfg.CASE_COLORS[case], hatch=hatch, edgecolor="white", lw=0,
                   label=f"{cfg.CASES[case]['short']} - {'immagine' if comp == 'phi_img' else 'clinica'}")
            for p, v in zip(pos, vals):
                a1.text(p, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
            k += 1
    a1.set_xticks(x, labels)
    a1.set_ylabel("media di |SHAP| (log-odds)")
    a1.set_title("Contributo per modalità")
    a1.grid(axis="x", visible=False)
    a1.legend(frameon=False, fontsize=7, loc="upper left")
    # Destra: PR-AUC con ciascuna modalità marginalizzata (media ± ds sui fold).
    metrics = [("prauc_full", "modello completo"), ("prauc_tab_marginalized", "senza clinica"),
               ("prauc_img_marginalized", "senza immagine")]
    x = np.arange(len(metrics))
    w = 0.8 / len(cases)
    for k, case in enumerate(cases):
        f = folds[case]
        means = [f[m].mean() for m, _ in metrics]
        stds = [f[m].std(ddof=1) for m, _ in metrics]
        pos = x + (k - (len(cases) - 1) / 2) * w
        a2.bar(pos, means, yerr=stds, width=w * 0.9, color=cfg.CASE_COLORS[case],
               error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2}, label=cfg.CASES[case]["short"])
        for p, m, s in zip(pos, means, stds):
            a2.text(p, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    prev = data[cases[0]]["target"].mean()
    a2.axhline(prev, color=MUTED, lw=0.8, ls="--", label=f"classificatore casuale ({prev:.3f})")
    a2.set_xticks(x, [lab for _, lab in metrics])
    a2.set_ylabel("PR-AUC (media sui 5 fold)")
    a2.set_title("Ablazione per marginalizzazione")
    a2.grid(axis="x", visible=False)
    a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.tight_layout()
    save(fig, "shap_modalita.png")


def select_cases(d):
    """Vero positivo più sicuro, falso negativo più netto, falso positivo più netto."""
    pos, neg = d[d.target == 1], d[d.target == 0]
    return {
        "vero_positivo": pos.loc[pos["prob"].idxmax(), "isic_id"],
        "falso_negativo": pos.loc[pos["prob"].idxmin(), "isic_id"],
        "falso_positivo": neg.loc[neg["prob"].idxmax(), "isic_id"],
    }


def _fmt_value(var, val):
    if pd.isna(val):
        return "mancante"
    if var in cfg.CAT_VARS:
        return cfg.CAT_LABELS.get(val, val)
    if var == "age_approx":
        return f"{val:.0f} anni"
    if var == "tbp_lv_areaMM2":
        return f"{val:.1f} mm²"
    return f"{val:.2f}"


def waterfall(ax, row, color):
    phis = np.array([row[f"phi_{v}"] for v in cfg.VAR_ORDER])
    order = np.argsort(np.abs(phis))
    start = row["base_tab"]
    ypos = np.arange(len(order))
    lefts = start + np.cumsum(phis[order]) - phis[order]
    rights = lefts + phis[order]
    lo, hi = min(lefts.min(), rights.min(), start), max(lefts.max(), rights.max(), start)
    span = max(hi - lo, 1e-3)
    ax.set_xlim(lo - 0.08 * span, hi + 0.22 * span)
    for y, i, left in zip(ypos, order, lefts):
        val = phis[i]
        ax.barh(y, val, left=left, height=0.6, color="#c0392b" if val > 0 else "#2e6fb0", alpha=0.85)
        ax.text(max(left, left + val) + 0.02 * span, y, f"{val:+.2f}", va="center", fontsize=7, color=INK)
    labels = [f"{cfg.VAR_LABELS[cfg.VAR_ORDER[i]]} = {_fmt_value(cfg.VAR_ORDER[i], row[f'val_{cfg.VAR_ORDER[i]}'])}"
              for i in order]
    ax.set_yticks(ypos, labels, fontsize=7)
    ax.axvline(start, color=MUTED, lw=0.8, ls="--")
    ax.axvline(row["logit"], color=color, lw=1.4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel(f"log-odds di malignità  |  base (clinica marginalizzata) {start:.2f} → f(x) {row['logit']:.2f}"
                  f"  (P = {row['prob']:.3f})", fontsize=7, color=MUTED)


def fig_waterfalls(data):
    cases = list(data)
    chosen = select_cases(data[cases[0]])
    names = {"vero_positivo": "vero positivo", "falso_negativo": "falso negativo", "falso_positivo": "falso positivo"}
    fig, axes = plt.subplots(len(chosen), len(cases), figsize=(5.6 * len(cases), 2.9 * len(chosen)), squeeze=False)
    for r, (kind, isic) in enumerate(chosen.items()):
        for c, case in enumerate(cases):
            row = data[case].set_index("isic_id").loc[isic]
            ax = axes[r, c]
            waterfall(ax, row, cfg.CASE_COLORS[case])
            truth = "maligna" if row["target"] == 1 else "benigna"
            ax.set_title(f"{cfg.CASES[case]['short']} | {isic} ({truth}, {names[kind]} per il Caso 3)\n"
                         f"SHAP immagine = {row['phi_img']:+.2f}, clinica = {row['phi_tab']:+.2f}", fontsize=8)
    fig.tight_layout()
    save(fig, "shap_waterfall_casi.png")
    return chosen


def stability(data):
    """Correlazione di Spearman tra i ranking delle variabili nei diversi fold."""
    out = {}
    for case, d in data.items():
        per_fold = d.groupby("fold")[[f"phi_{v}" for v in cfg.VAR_ORDER]].apply(lambda x: x.abs().mean())
        ranks = per_fold.rank(axis=1)
        corr = ranks.T.corr(method="pearson").to_numpy()
        iu = np.triu_indices_from(corr, k=1)
        out[case] = {"spearman_mean": float(corr[iu].mean()), "spearman_min": float(corr[iu].min()),
                     "top_variable_per_fold": per_fold.idxmax(axis=1).str.replace("phi_", "").tolist()}
    return out


def main():
    data = {c: d for c in cfg.CASES if (d := load(c)) is not None}
    if not data:
        raise SystemExit(f"Nessun risultato in {cfg.OUT_DIR}: esegui prima python -m XAI_SHAP.shap_run")
    folds = {c: pd.read_csv(os.path.join(cfg.OUT_DIR, f"fold_summary_{c}.csv")) for c in data}
    print("Figure:")
    imp = importance_table(data)
    fig_importance(data, imp)
    fig_beeswarm(data, imp)
    fig_dependence(data)
    fig_categorical(data)
    fig_modality(data, folds)
    chosen = fig_waterfalls(data)

    # Tabelle e riepilogo testuale.
    imp.to_csv(os.path.join(cfg.OUT_DIR, "tabella_importanza.csv"), index=False)
    lines = ["RIEPILOGO SHAP - ISIC 2024 (output spiegato: log-odds di malignità)", ""]
    for case, d in data.items():
        f = folds[case]
        share = d["phi_tab"].abs().mean() / (d["phi_tab"].abs().mean() + d["phi_img"].abs().mean())
        lines += [f"== {cfg.CASES[case]['label']} ==",
                  f"  PR-AUC completo         {f.prauc_full.mean():.4f} ± {f.prauc_full.std(ddof=1):.4f}"
                  f"   (CSV, epoca migliore: {f.prauc_csv_best_epoch.mean():.4f})",
                  f"  PR-AUC senza clinica    {f.prauc_tab_marginalized.mean():.4f} ± {f.prauc_tab_marginalized.std(ddof=1):.4f}",
                  f"  PR-AUC senza immagine   {f.prauc_img_marginalized.mean():.4f} ± {f.prauc_img_marginalized.std(ddof=1):.4f}",
                  f"  |SHAP| medio immagine   {d.phi_img.abs().mean():.4f}   clinica {d.phi_tab.abs().mean():.4f}"
                  f"   (quota clinica {share:.1%})",
                  f"  Decisioni a soglia 0.5 (senza clinica -> con clinica): maligne individuate "
                  f"{((d.base_tab >= 0) & (d.target == 1)).sum()} -> {((d.logit >= 0) & (d.target == 1)).sum()} su {int(d.target.sum())}, "
                  f"falsi positivi {((d.base_tab >= 0) & (d.target == 0)).sum()} -> {((d.logit >= 0) & (d.target == 0)).sum()}",
                  f"  Errore max proiezione {f.max_err_projection.max():.1e} | efficienza {f.max_err_efficiency.max():.1e}",
                  "  Importanza (media |SHAP| ± ds tra fold):"]
        for _, r in imp[imp.case == case].sort_values("mean_abs_phi", ascending=False).iterrows():
            lines.append(f"    {cfg.VAR_LABELS[r.variable]:22s} {r.mean_abs_phi:.4f} ± {r.std_between_folds:.4f}")
        lines.append("")
    stab = stability(data)
    lines.append("Stabilità del ranking tra fold (Spearman medio / minimo):")
    for case, s in stab.items():
        lines.append(f"  {cfg.CASES[case]['short']:20s} {s['spearman_mean']:.3f} / {s['spearman_min']:.3f}"
                     f"   variabile principale per fold: {s['top_variable_per_fold']}")
    lines += ["", f"Casi nei waterfall: {chosen}"]
    text = "\n".join(lines)
    with open(os.path.join(cfg.OUT_DIR, "riepilogo_shap.txt"), "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
