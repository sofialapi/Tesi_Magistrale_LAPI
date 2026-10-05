"""Figura per la tesi: immagine + due mappe Grad-CAM++ + spiegazione sotto.

Layout (16 pollici di larghezza, testo allineato sotto il pannello a cui si riferisce):

  [ immagine ]        [ ResNet-50 + CAM ]   [ MobileViT-S + CAM ]
  Sintesi             ResNet-50             MobileViT-S
  ---------------------------------------------------------------
  Confronto tra i modelli         |  Avvertenze per il clinico
  nota a pie' di figura
"""
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

FIG_W = 16.0
MARGIN = 0.35
GAP = 0.35
FS = 10.0         # corpo del testo
HFS = 11.0        # intestazioni
CHAR_EM = 0.49    # larghezza media di un carattere in em (stima prudente)
LINESPACING = 1.35
LINE = FS * 1.22 * LINESPACING / 72  # altezza reale di una riga in matplotlib
HEAD = HFS * 1.75 / 72
WARN_COLOR = "#b45309"
HEAD_COLOR = "#1f2937"
BODY_COLOR = "#111827"
MUTED = "#6b7280"


def overlay_cam(rgb_uint8, cam, image_weight=0.5):
  """Equivalente a pytorch_grad_cam.show_cam_on_image (colormap JET)."""
  heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
  heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  img = rgb_uint8.astype(np.float32) / 255.0
  out = (1 - image_weight) * heat + image_weight * img
  return out / max(float(out.max()), 1e-8)


def _wrap(text, width_in, fs=FS, bullet=False):
  chars = max(20, int(width_in * 72 / (fs * CHAR_EM)))
  if bullet:
    return textwrap.wrap(text, chars, initial_indent="•  ",
                         subsequent_indent="    ")
  lines = []
  for par in str(text).split("\n"):
    lines += textwrap.wrap(par, chars) or [""]
  return lines


def _block(heading, lines, color=HEAD_COLOR):
  return {"heading": heading, "lines": lines, "color": color,
          "height": HEAD + len(lines) * LINE}


def _draw_block(fig, H, x_in, y_in, block):
  fig.text(x_in / FIG_W, 1 - y_in / H, block["heading"], ha="left", va="top",
           fontsize=HFS, fontweight="bold", color=block["color"])
  fig.text(x_in / FIG_W, 1 - (y_in + HEAD) / H, "\n".join(block["lines"]),
           ha="left", va="top", fontsize=FS, color=BODY_COLOR,
           linespacing=LINESPACING)


def _hline(fig, H, y_in):
  y = 1 - y_in / H
  fig.add_artist(Line2D([MARGIN / FIG_W, 1 - MARGIN / FIG_W], [y, y],
                        transform=fig.transFigure, color="#d1d5db", lw=0.8))


def render_case_figure(out_path, panel_rgb, overlay_rgb, cams, mask, info,
                       sections, extra_warnings=(), footer="", dpi=300,
                       show_contour=True):
  """Salva la figura del caso.

  panel_rgb:   immagine del primo pannello (originale, se disponibile)
  overlay_rgb: immagine su cui sovrapporre le mappe (quella vista dal modello)
  cams:        {"resnet50": cam, "mobilevit_s": cam}
  info:        id, ground_truth, dataset, soglia, predizioni per modello
  sections:    sintesi, resnet50, mobilevit_s, confronto, avvertenze (lista)
  """
  pw = (FIG_W - 2 * MARGIN - 2 * GAP) / 3
  ph = pw * panel_rgb.shape[0] / panel_rgb.shape[1]  # rispetta le proporzioni
  xs = [MARGIN + i * (pw + GAP) for i in range(3)]

  # ---- testo: misura dei blocchi ----
  row1 = [
      _block("Sintesi", _wrap(sections["sintesi"], pw)),
      _block("ResNet-50: regioni rilevanti", _wrap(sections["resnet50"], pw)),
      _block("MobileViT-S: regioni rilevanti", _wrap(sections["mobilevit_s"], pw)),
  ]
  half = (FIG_W - 2 * MARGIN - GAP) / 2
  warns = list(sections.get("avvertenze") or []) + [
      f"{w} (aggiunta automaticamente)" for w in extra_warnings]
  if warns:
    wlines = []
    for w in warns:
      wlines += _wrap(w, half, bullet=True)
    wblock = _block("Avvertenze per il clinico", wlines, WARN_COLOR)
  else:
    wblock = _block("Avvertenze per il clinico",
                    ["Nessuna avvertenza rilevata dai controlli automatici."])
  row2 = [_block("Confronto tra i modelli", _wrap(sections["confronto"], half)),
          wblock]
  foot_lines = _wrap(footer, FIG_W - 2 * MARGIN, fs=7.5) if footer else []

  title_h, top, gap1, gap2, gap3, bottom = 0.95, 0.15, 0.30, 0.28, 0.22, 0.18
  h_row1 = max(b["height"] for b in row1)
  h_row2 = max(b["height"] for b in row2)
  h_foot = len(foot_lines) * 7.5 * 1.4 / 72
  H = top + title_h + ph + gap1 + h_row1 + gap2 + h_row2 + gap3 + h_foot + bottom

  fig = plt.figure(figsize=(FIG_W, H), facecolor="white")

  # ---- pannelli immagine ----
  y_img = top + title_h
  tau = info["soglia"]
  titles = [
      (f"Immagine: {info['image_id']}\nGround Truth: {info['ground_truth']}",
       "black", "bold"),
  ]
  for k, name in (("resnet50", "ResNet-50 (CNN)"), ("mobilevit_s", "MobileViT-S (Hybrid)")):
    p = info["prob"][k]
    esito = info["esito"][k]
    color = "green" if esito == info["ground_truth"] else "red"
    titles.append((f"{name}\nP(Melanoma): {p:.4f}\nEsito (tau={tau}): {esito}",
                   color, "normal"))
  images = [panel_rgb,
            overlay_cam(overlay_rgb, cams["resnet50"]),
            overlay_cam(overlay_rgb, cams["mobilevit_s"])]

  for i in range(3):
    ax = fig.add_axes([xs[i] / FIG_W, 1 - (y_img + ph) / H, pw / FIG_W, ph / H])
    ax.imshow(images[i])
    if show_contour and mask is not None and mask.any():
      ax.contour(mask.astype(float), levels=[0.5], colors="white",
                 linewidths=1.0, linestyles="--")
    t, c, wgt = titles[i]
    ax.set_title(t, fontsize=12, color=c, fontweight=wgt, pad=8)
    ax.axis("off")

  # ---- testo ----
  y = y_img + ph + gap1
  _hline(fig, H, y - gap1 / 2)
  for i, b in enumerate(row1):
    _draw_block(fig, H, xs[i], y, b)
  y += h_row1 + gap2
  _hline(fig, H, y - gap2 / 2)
  _draw_block(fig, H, MARGIN, y, row2[0])
  _draw_block(fig, H, MARGIN + half + GAP, y, row2[1])
  y += h_row2 + gap3
  if foot_lines:
    fig.text(MARGIN / FIG_W, 1 - y / H, "\n".join(foot_lines), ha="left",
             va="top", fontsize=7.5, color=MUTED, style="italic",
             linespacing=1.4)

  fig.savefig(out_path, dpi=dpi, facecolor="white")
  plt.close(fig)
  return out_path
