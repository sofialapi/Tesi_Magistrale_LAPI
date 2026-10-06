"""Scheda per il clinico (PNG): immagine, Grad-CAM++, percorso della decisione,
contributi dei dati clinici e spiegazione testuale.

Layout (16 pollici di larghezza, come le figure della Sezione 6.2):

  [ crop TBP ] [ Grad-CAM++ ] [ percorso della decisione ] [ dati clinici ]
  Sintesi ----------------------------------------------------------------
  Immagine                      | Dati clinici
  Avvertenze per il clinico ----------------------------------------------
  nota a pie' di figura
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from XAI_NL.nl_figure import (FIG_W, GAP, HEAD, LINE, MARGIN, WARN_COLOR, _block,  # noqa: E402
                              _draw_block, _hline, _wrap, overlay_cam)

POS, NEG, NEUTRAL, INK, MUTED = "#c0392b", "#2e6fb0", "#8a8a8a", "#222222", "#6b7280"


def _logit(p):
  p = min(max(p, 1e-3), 1 - 1e-3)
  return math.log(p / (1 - p))


def _decision_path(ax, prob, tau, color):
  """Riferimento -> sola immagine -> immagine + dati clinici, su scala logit."""
  steps = [("Lesione di\nriferimento", prob["riferimento"]),
           ("Sola\nimmagine", prob["solo_immagine"]),
           ("Immagine +\ndati clinici", prob["immagine_e_dati_clinici"])]
  ticks = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
  ax.set_xlim(_logit(0.004), _logit(0.996))
  ax.set_ylim(-0.75, 2.45)
  ax.invert_yaxis()
  ax.axvline(_logit(tau), color=MUTED, ls="--", lw=1)
  ax.text(_logit(tau), -0.72, f"soglia {tau}", ha="center", va="bottom", fontsize=8, color=MUTED)
  xs = [_logit(p) for _, p in steps]
  for i, (lab, p) in enumerate(steps):
    if i > 0:
      c = color if i == 1 else (POS if xs[2] > xs[1] else NEG)
      ax.annotate("", xy=(xs[i], i), xytext=(xs[i - 1], i),
                  arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, mutation_scale=14))
      ax.plot([xs[i - 1], xs[i - 1]], [i - 1, i], color="#d1d5db", lw=0.8, zorder=0)
    ax.plot(xs[i], i, "o", ms=8, color=NEUTRAL if i == 0 else (color if i == 1 else INK), zorder=3)
    ax.text(xs[i], i - 0.32, f"{p:.3f}", fontsize=9, ha="center", va="bottom", color=INK,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
  ax.set_yticks(range(3), [s for s, _ in steps], fontsize=8.5)
  ax.set_xticks([_logit(t) for t in ticks], [f"{t:g}" for t in ticks], fontsize=8)
  ax.set_xlabel("probabilità di malignità (scala logistica)", fontsize=8.5)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  ax.grid(axis="x", color="#eeeeee", lw=0.6)
  ax.set_axisbelow(True)


def _factors(ax, fattori):
  items = list(reversed(fattori))  # piu' importante in alto
  phis = np.array([x["phi"] for x in items])
  y = np.arange(len(items))
  ax.barh(y, phis, color=[POS if v > 0 else NEG for v in phis], height=0.62, alpha=0.9)
  lim = max(0.3, float(np.abs(phis).max()) * 1.35)
  ax.set_xlim(-lim, lim)
  ax.axvline(0, color=MUTED, lw=0.8)
  for yi, v in zip(y, phis):
    ax.text(v + (0.03 * lim if v >= 0 else -0.03 * lim), yi, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8, color=INK)
  labels = [f"{x['nome_breve']}: {x['valore_breve']}" for x in items]
  ax.set_yticks(y, labels, fontsize=8)
  ax.set_xticks([])
  ax.set_xlabel("← verso benignità        verso malignità →", fontsize=8.5)
  for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)


def render_card(out_path, crop_rgb, model_rgb, cam, mask, info, features, sections,
                extra_warnings=(), footer="", dpi=300):
  """info: image_id, verita' ('Maligna'/'Benigna'), titolo_modello, colore_modello."""
  tau = features["soglia_decisionale"]
  sq = 2.95                       # lato dei pannelli immagine (pollici)
  w_path = 4.6
  w_fac = FIG_W - 2 * MARGIN - 2 * sq - w_path - 3 * GAP
  lab_w = 1.9                     # spazio per le etichette del grafico dei fattori
  xs = [MARGIN, MARGIN + sq + GAP, MARGIN + 2 * (sq + GAP)]
  x_fac = xs[2] + w_path + GAP

  full = FIG_W - 2 * MARGIN
  half = (full - GAP) / 2
  b_sint = _block("Sintesi", _wrap(sections["sintesi"], full))
  b_img = _block("Immagine", _wrap(sections["immagine"], half))
  b_cli = _block("Dati clinici", _wrap(sections["dati_clinici"], half))
  warns = list(sections.get("avvertenze") or []) + [
      f"{w} (aggiunta automaticamente)" for w in extra_warnings]
  if warns:
    wl = []
    for w in warns:
      wl += _wrap(w, full, bullet=True)
    b_warn = _block("Avvertenze per il clinico", wl, WARN_COLOR)
  else:
    b_warn = _block("Avvertenze per il clinico",
                    ["Nessuna avvertenza rilevata dai controlli automatici."])
  foot = _wrap(footer, full, fs=7.5) if footer else []

  title_h, top, gap1, gap2, bottom = 0.55, 0.15, 0.45, 0.25, 0.18
  h_foot = len(foot) * 7.5 * 1.4 / 72
  y_pan = top + title_h + 0.55
  H = (y_pan + sq + gap1 + b_sint["height"] + gap2 + max(b_img["height"], b_cli["height"])
       + gap2 + b_warn["height"] + gap2 + h_foot + bottom)
  fig = plt.figure(figsize=(FIG_W, H), facecolor="white")

  p = features["probabilita"]["immagine_e_dati_clinici"]
  esito = features["esito"]
  ok = esito.lower() == info["verita"].lower()
  fig.text(MARGIN / FIG_W, 1 - top / H,
           f"Scheda di spiegazione  |  {info['image_id']}  |  {features['modello']}",
           fontsize=14, fontweight="bold", va="top", color=INK)
  fig.text(MARGIN / FIG_W, 1 - (top + 0.32) / H,
           f"Esito: lesione {esito} (P = {p:.3f}, soglia {tau})   ·   verità: "
           f"{info['verita'].lower()} (non fornita al generatore di testo)",
           fontsize=11, va="top", color="#15803d" if ok else "#b91c1c")

  def img_ax(x, image, title):
    ax = fig.add_axes([x / FIG_W, 1 - (y_pan + sq) / H, sq / FIG_W, sq / H])
    ax.imshow(image)
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")
    return ax

  img_ax(xs[0], crop_rgb, "Crop TBP originale")
  ax = img_ax(xs[1], overlay_cam(model_rgb, cam),
              "Grad-CAM++ sul ramo visivo\n(bassa risoluzione: zona indicativa)")
  if mask is not None and mask.any():
    ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=1.0, linestyles="--")

  pad_l = 0.95
  axp = fig.add_axes([(xs[2] + pad_l) / FIG_W, 1 - (y_pan + sq - 0.45) / H,
                      (w_path - pad_l) / FIG_W, (sq - 0.75) / H])
  _decision_path(axp, features["probabilita"], tau, info["colore_modello"])
  axp.set_title("Percorso della decisione", fontsize=10, pad=14)

  axf = fig.add_axes([(x_fac + lab_w) / FIG_W, 1 - (y_pan + sq - 0.45) / H,
                      (w_fac - lab_w) / FIG_W, (sq - 0.75) / H])
  _factors(axf, features["fattori_clinici"])
  axf.set_title("Contributo dei dati clinici", fontsize=10, pad=14)

  y = y_pan + sq + gap1
  _hline(fig, H, y - gap1 / 2 + 0.12)
  _draw_block(fig, H, MARGIN, y, b_sint)
  y += b_sint["height"] + gap2
  _draw_block(fig, H, MARGIN, y, b_img)
  _draw_block(fig, H, MARGIN + half + GAP, y, b_cli)
  y += max(b_img["height"], b_cli["height"]) + gap2
  _hline(fig, H, y - gap2 / 2)
  _draw_block(fig, H, MARGIN, y, b_warn)
  y += b_warn["height"] + gap2
  if foot:
    fig.text(MARGIN / FIG_W, 1 - y / H, "\n".join(foot), ha="left", va="top", fontsize=7.5,
             color=MUTED, style="italic", linespacing=1.4)
  fig.savefig(out_path, dpi=dpi, facecolor="white")
  plt.close(fig)
  return out_path
