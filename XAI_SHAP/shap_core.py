"""Valori di Shapley esatti (enumerazione completa delle coalizioni), solo numpy.

Il modello è visto come una funzione  f(c, t) -> logit, dove
  c = contesto visivo (proiezione dell'embedding dell'immagine),
  t = vettore clinico preelaborato (11 colonne).

SHAP interventional: una variabile "assente" prende il valore di un campione di
background, e il valore della coalizione è la media su tutto il background.
"""
from itertools import combinations
from math import factorial

import numpy as np


def coalition_masks(n_players: int) -> np.ndarray:
    """Tutte le 2^n coalizioni come maschere booleane (2^n, n), ordinate per dimensione."""
    masks = []
    for k in range(n_players + 1):
        for subset in combinations(range(n_players), k):
            m = np.zeros(n_players, dtype=bool)
            m[list(subset)] = True
            masks.append(m)
    return np.array(masks)


def shapley_matrix(masks: np.ndarray) -> np.ndarray:
    """Matrice W (n, 2^n) tale che phi = v @ W.T, con v i valori delle coalizioni.

    phi_i = sum_{S non contiene i} |S|!(n-|S|-1)!/n! * [v(S u {i}) - v(S)]
    Il coefficiente di v(S) è +w(|S|-1) se i in S, -w(|S|) se i non in S.
    """
    n = masks.shape[1]
    sizes = masks.sum(axis=1)
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    W = np.zeros((n, len(masks)))
    for i in range(n):
        inside = masks[:, i]
        W[i, inside] = w[sizes[inside] - 1]
        W[i, ~inside] = -w[sizes[~inside]]
    return W


def tabular_coalition_values(predict, ctx, tab, background, groups, masks=None, chunk=16):
    """Valori v(S) per ogni campione e coalizione di variabili cliniche.

    predict    : callable(ctx_rows (R, C), tab_rows (R, F)) -> logit (R,)
    ctx        : (N, C) contesto visivo dei campioni da spiegare (fisso)
    tab        : (N, F) vettori clinici dei campioni da spiegare
    background : (B, F) vettori clinici di background
    groups     : lista di array di indici di colonna, uno per giocatore
    Ritorna (values (N, 2^G), masks (2^G, G)).
    """
    n_groups = len(groups)
    if masks is None:
        masks = coalition_masks(n_groups)
    n_feat = tab.shape[1]
    col_masks = np.zeros((len(masks), n_feat), dtype=bool)
    for g, cols in enumerate(groups):
        col_masks[:, cols] = masks[:, [g]]

    n, n_coal, n_bg = len(tab), len(masks), len(background)
    values = np.empty((n, n_coal))
    for start in range(0, n, chunk):
        t = tab[start:start + chunk]                      # (n_c, F)
        c = ctx[start:start + chunk]                      # (n_c, C)
        n_c = len(t)
        rows = np.where(col_masks[None, :, None, :], t[:, None, None, :], background[None, None, :, :])
        ctx_rows = np.broadcast_to(c[:, None, None, :], (n_c, n_coal, n_bg, c.shape[1]))
        out = predict(ctx_rows.reshape(-1, c.shape[1]), rows.reshape(-1, n_feat))
        values[start:start + n_c] = np.asarray(out).reshape(n_c, n_coal, n_bg).mean(axis=2)
    return values, masks


def modality_shapley(v_full, v_img_only, v_tab_only, v_none):
    """Shapley a due giocatori (immagine, clinica).

    v_full     = f(img_x, tab_x)
    v_img_only = E_b f(img_x, tab_b)   (clinica marginalizzata)
    v_tab_only = E_b f(img_b, tab_x)   (immagine marginalizzata)
    v_none     = E_b f(img_b, tab_b)
    """
    phi_img = 0.5 * ((v_img_only - v_none) + (v_full - v_tab_only))
    phi_tab = 0.5 * ((v_tab_only - v_none) + (v_full - v_img_only))
    return phi_img, phi_tab


def build_groups(feature_names, var_order):
    """Indici di colonna per ciascuna variabile originale (one-hot raggruppate)."""
    groups = []
    for var in var_order:
        idx = [i for i, name in enumerate(feature_names) if name == var or name.startswith(var + "_")]
        if not idx:
            raise ValueError(f"Nessuna colonna trovata per la variabile '{var}'")
        groups.append(np.array(idx))
    covered = np.sort(np.concatenate(groups))
    if not np.array_equal(covered, np.arange(len(feature_names))):
        raise ValueError(f"I gruppi non coprono esattamente le colonne: {feature_names}")
    return groups
