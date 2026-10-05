"""Autotest del modulo SHAP (non usa dati reali né checkpoint; gira anche su CPU).

Uso: python -m XAI_SHAP.shap_selftest

Verifica che:
 1. HeadFunction (testa scomposta) riproduca il forward completo di DermalClassifier;
 2. gli Shapley esatti di shap_core coincidano con shap.explainers.Exact
    (libreria SHAP ufficiale, masker Independent) sulle 11 colonne;
 3. valga l'efficienza (somma dei contributi = f(x) - base) con le 6 variabili raggruppate;
 4. il raggruppamento one-hot sia corretto sui nomi prodotti dal processor reale.
"""
import numpy as np
import pandas as pd
import shap
import torch

from XAI_SHAP import shap_config as cfg
from XAI_SHAP.shap_core import build_groups, shapley_matrix, tabular_coalition_values
from XAI_SHAP.shap_data import DermalClassifier, HeadFunction, fit_processor

TOL = 1e-4


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'ERRORE'}] {name} {detail}")
    if not ok:
        raise SystemExit(1)


def synthetic_metadata(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "isic_id": [f"ISIC_{i:07d}" for i in range(n)],
        "target": (rng.random(n) < 0.05).astype(int),
        "age_approx": np.where(rng.random(n) < 0.02, np.nan, rng.integers(3, 18, n) * 5.0),
        "tbp_lv_symm_2axis": rng.uniform(0.07, 0.9, n),
        "tbp_lv_eccentricity": rng.uniform(0.1, 0.97, n),
        "tbp_lv_areaMM2": np.exp(rng.normal(1.8, 0.8, n)),
        "sex": rng.choice(["male", "female", None], n, p=[0.65, 0.32, 0.03]),
        "anatom_site_general": rng.choice(["posterior torso", "lower extremity", "anterior torso",
                                           "upper extremity", "head/neck"], n),
    })


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    print("4. Processor e raggruppamento delle variabili")
    df = synthetic_metadata()
    _, X, names = fit_processor(df)
    groups = build_groups(names, cfg.VAR_ORDER)
    check("colonne del processor", X.shape[1] == 11, f"({X.shape[1]}: {names})")
    check("6 gruppi che coprono le 11 colonne", len(groups) == 6, str([list(g) for g in groups]))

    rng = np.random.default_rng(1)
    for backbone in ("cnn", "hybrid"):
        print(f"\n[{backbone}] modello con pesi casuali")
        model = DermalClassifier(backbone_type=backbone, mode="multimodal", clinical_dim=11, pretrained=False).eval()
        # statistiche BatchNorm non banali, per rendere il test significativo
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_(0, 0.5)
                m.running_var.uniform_(0.5, 2.0)
        head = HeadFunction(model, "cpu")
        imgs = torch.randn(3, 3, 224, 224)
        clin = X[:3]
        bg = X[rng.choice(len(X), 40, replace=False)]

        z = model(imgs, torch.as_tensor(clin)).numpy()
        ctx = head.project(model.visual_backbone(imgs).numpy())
        err = np.abs(head.predict(ctx, clin) - (z[:, 1] - z[:, 0])).max()
        check("1. testa scomposta = forward completo", err < TOL, f"(errore max {err:.1e})")

        singles = [np.array([j]) for j in range(11)]
        vals, masks = tabular_coalition_values(head.predict, ctx, clin, bg, singles, chunk=1)
        phi_ours = vals @ shapley_matrix(masks).T
        phi_lib = np.zeros_like(phi_ours)
        for i in range(len(clin)):
            f = lambda T, i=i: head.predict(np.repeat(ctx[i:i + 1], len(T), axis=0), T)
            explainer = shap.explainers.Exact(f, shap.maskers.Independent(bg, max_samples=len(bg)))
            phi_lib[i] = explainer(clin[i:i + 1]).values[0]
        err = np.abs(phi_ours - phi_lib).max()
        check("2. Shapley esatti = shap.explainers.Exact", err < TOL,
              f"(errore max {err:.1e}, |phi| max {np.abs(phi_lib).max():.2f})")

        vals6, masks6 = tabular_coalition_values(head.predict, ctx, clin, bg, groups)
        phi6 = vals6 @ shapley_matrix(masks6).T
        err = np.abs(phi6.sum(1) - (vals6[:, -1] - vals6[:, 0])).max()
        check("3. efficienza con 6 variabili raggruppate", err < 1e-8, f"(errore max {err:.1e})")

    print("\nTutti i controlli superati.")


if __name__ == "__main__":
    main()
