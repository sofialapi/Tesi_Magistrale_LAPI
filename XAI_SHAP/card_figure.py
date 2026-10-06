"""Scheda per il clinico (PNG verticale, pensata per la larghezza di una pagina A4).

Larga 8.3 pollici: inserita a \\textwidth (circa 15-16 cm) il testo resta intorno
ai 7 pt, quindi leggibile senza ruotare la pagina.

  Titolo / esito
  [ crop TBP ]                 [ Grad-CAM++ ]
  [ percorso della decisione ] [ contributo dei dati clinici ]
  Sintesi ----------------------------------------------------
  Immagine                     | Dati clinici
  Avvertenze per il clinico ----------------------------------
  nota a pie' di figura
"""
import math
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from XAI_NL.nl_figure import overlay_cam  # noqa: E402

FIG_W = 8.3
MARGIN = 0.30
GAP = 0.30
COL = (FIG_W - 2 * MARGIN - GAP) / 2
FS, HFS, FOOT_FS = 9.5, 10.5, 7.5
CHAR_EM = 0.50
LINESPACING = 1.3
POS, NEG, NEUTRAL, INK, MUTED = "#c0392b", "#2e6fb0", "#8a8a8a", "#222222", "#6b7280"
HEAD_COLOR, WARN_COLOR = "#1f2937", "#b45309"


def _line_h(fs):
  return fs * 1.22 * LINESPACING / 72


def _wrap(text, width_in, fs=FS, bullet=False):
  chars = max(20, int(width_in * 72 / (fs * CHAR_EM)))
  if bullet:
    return textwrap.wrap(text, chars, initial_indent="•  ", subsequent_indent="    ")
  out = []
  for par in str(text).split("\n"):
    out += textwrap.wrap(par, chars) or [""]
  return out


def _block(heading, lines, color=HEAD_COLOR):
  return {"heading": heading, "lines": lines, "color": color,
          "height": HFS * 1.7 / 72 + len(lines) * _line_h(FS)}


def _draw_block(fig, H, x, y, b):
  fig.text(x / FIG_W, 1 - y / H, b["heading"], ha="left", va="top", fontsize=HFS,
           fontweight="bold", color=b["color"])
  fig.text(x / FIG_W, 1 - (y + HFS * 1.7 / 72) / H, "\n".join(b["lines"]), ha="left",
           va="top", fontsize=FS, color=INK, linespacing=LINESPACING)


def _hline(fig, H, y):
  fig.add_artist(Line2D([MARGIN / FIG_W, 1 - MARGIN / FIG_W], [1 - y / H] * 2,
                        transform=fig.transFigure, color="#d1d5db", lw=0.8))


def _logit(p):
  p = min(max(p, 1e-3), 1 - 1e-3)
  return math.log(p / (1 - p))


def _decision_path(ax, prob, tau, color):
  """Riferimento -> sola immagine -> immagine + dati clinici, su scala logistica."""
  steps = [("Lesione di\nriferimento", prob["riferimento"]),
           ("Sola\nimmagine", prob["solo_immagine"]),
           ("Immagine +\ndati clinici", prob["immagine_e_dati_clinici"])]
  ticks = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
  ax.set_xlim(_logit(0.004), _logit(0.996))
  ax.set_ylim(-0.75, 2.45)
  ax.invert_yaxis()
  ax.axvline(_logit(tau), color=MUTED, ls="--", lw=1)
  ax.text(_logit(tau), -0.72, f"soglia {tau}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
  xs = [_logit(p) for _, p in steps]
  for i, (_, p) in enumerate(steps):
    if i > 0:
      c = color if i == 1 else (POS if xs[2] > xs[1] else NEG)
      ax.annotate("", xy=(xs[i], i), xytext=(xs[i - 1], i),
                  arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=12))
      ax.plot([xs[i - 1]] * 2, [i - 1, i], color="#d1d5db", lw=0.8, zorder=0)
    ax.plot(xs[i], i, "o", ms=7, color=NEUTRAL if i == 0 else (color if i == 1 else INK), zorder=3)
    ax.text(xs[i], i - 0.3, f"{p:.3f}", fontsize=8.5, ha="center", va="bottom", color=INK,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
  ax.set_yticks(range(3), [s for s, _ in steps], fontsize=8)
  ax.set_xticks([_logit(t) for t in ticks], [f"{t:g}" for t in ticks], fontsize=7.5)
  ax.set_xlabel("probabilità di malignità (scala logistica)", fontsize=8)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  ax.grid(axis="x", color="#eeeeee", lw=0.6)
  ax.set_axisbelow(True)


def _factors(ax, fattori):
  items = list(reversed(fattori))  # piu' importante in alto
  phis = np.array([x["phi"] for x in items])
  y = np.arange(len(items))
  ax.barh(y, phis, color=[POS if v > 0 else NEG for v in phis], height=0.62, alpha=0.9)
  lim = max(0.3, float(np.abs(phis).max()) * 1.45)
  ax.set_xlim(-lim, lim)
  ax.axvline(0, color=MUTED, lw=0.8)
  for yi, v in zip(y, phis):
    ax.text(v + (0.04 * lim if v >= 0 else -0.04 * lim), yi, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
  ax.set_yticks(y, [f"{x['nome_breve']}: {x['valore_breve']}" for x in items], fontsize=7.5)
  ax.set_xticks([])
  ax.set_xlabel("← verso benignità   verso malignità →", fontsize=8)
  for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
  ax.tick_params(axis="y", length=0)


def render_card(out_path, crop_rgb, model_rgb, cam, mask, info, features, sections,
                extra_warnings=(), footer="", dpi=300):
  """info: image_id, verita' ('Maligna'/'Benigna'), colore_modello."""
  tau = features["soglia_decisionale"]
  full = FIG_W - 2 * MARGIN
  xl, xr = MARGIN, MARGIN + COL + GAP

  b_sint = _block("Sintesi", _wrap(sections["sintesi"], full))
  b_img = _block("Immagine", _wrap(sections["immagine"], COL))
  b_cli = _block("Dati clinici", _wrap(sections["dati_clinici"], COL))
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
  foot = _wrap(footer, full, fs=FOOT_FS) if footer else []
  h_foot = len(foot) * FOOT_FS * 1.4 / 72

  top, head_h = 0.15, 0.62
  img_t, img = 0.42, COL * 0.68           # titolo + lato delle immagini
  chart_t, chart_h, chart_b = 0.32, 1.85, 0.45
  gap = 0.22
  y_img = top + head_h + img_t
  y_chart = y_img + img + gap + chart_t
  y_text = y_chart + chart_h + chart_b + gap
  H = (y_text + b_sint["height"] + gap + max(b_img["height"], b_cli["height"]) + gap
       + b_warn["height"] + gap + h_foot + 0.15)
  fig = plt.figure(figsize=(FIG_W, H), facecolor="white")

  p = features["probabilita"]["immagine_e_dati_clinici"]
  esito = features["esito"]
  ok = esito.lower() == info["verita"].lower()
  fig.text(MARGIN / FIG_W, 1 - top / H,
           f"Scheda di spiegazione  |  {info['image_id']}  |  {features['modello']}",
           fontsize=11, fontweight="bold", va="top", color=INK)
  fig.text(MARGIN / FIG_W, 1 - (top + 0.28) / H,
           f"Esito: lesione {esito} (P = {p:.3f}, soglia {tau})  ·  verità: "
           f"{info['verita'].lower()} (non fornita al generatore di testo)",
           fontsize=9, va="top", color="#15803d" if ok else "#b91c1c")

  # immagini quadrate centrate nelle due colonne
  for x, image, title in ((xl, crop_rgb, "Crop TBP originale"),
                          (xr, overlay_cam(model_rgb, cam),
                           "Grad-CAM++ sul ramo visivo\n(bassa risoluzione: zona indicativa)")):
    xi = x + (COL - img) / 2
    ax = fig.add_axes([xi / FIG_W, 1 - (y_img + img) / H, img / FIG_W, img / H])
    ax.imshow(image)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.axis("off")
    if x == xr and mask is not None and mask.any():
      ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=0.9,
                 linestyles="--")

  lab_path, lab_fac = 0.85, 1.75
  axp = fig.add_axes([(xl + lab_path) / FIG_W, 1 - (y_chart + chart_h) / H,
                      (COL - lab_path) / FIG_W, chart_h / H])
  _decision_path(axp, features["probabilita"], tau, info["colore_modello"])
  axp.set_title("Percorso della decisione", fontsize=9, pad=12)
  axf = fig.add_axes([(xr + lab_fac) / FIG_W, 1 - (y_chart + chart_h) / H,
                      (COL - lab_fac) / FIG_W, chart_h / H])
  _factors(axf, features["fattori_clinici"])
  axf.set_title("Contributo dei dati clinici", fontsize=9, pad=12, x=0.25)

  y = y_text
  _hline(fig, H, y - gap / 2)
  _draw_block(fig, H, MARGIN, y, b_sint)
  y += b_sint["height"] + gap
  _draw_block(fig, H, xl, y, b_img)
  _draw_block(fig, H, xr, y, b_cli)
  y += max(b_img["height"], b_cli["height"]) + gap
  _hline(fig, H, y - gap / 2)
  _draw_block(fig, H, MARGIN, y, b_warn)
  y += b_warn["height"] + gap
  if foot:
    fig.text(MARGIN / FIG_W, 1 - y / H, "\n".join(foot), ha="left", va="top",
             fontsize=FOOT_FS, color=MUTED, style="italic", linespacing=1.4)
  fig.savefig(out_path, dpi=dpi, facecolor="white")
  plt.close(fig)
  return out_path
