"""Descrittori quantitativi da immagine preelaborata + mappe Grad-CAM++.

Questi descrittori sono l'unica fonte di informazione dell'LLM: ogni frase
della spiegazione deve poter essere ricondotta a un valore calcolato qui.
Sono indicatori euristici calcolati su 224x224 pixel, non una valutazione
dermoscopica (nessuna struttura come reticolo, strie o globuli e' misurata).
"""
import cv2
import numpy as np

from .nl_config import (CAM_HOT_THRESHOLD, MIN_LESION_ENERGY, MODEL_KEYS,
                        MODEL_NAMES, NEAR_THRESHOLD_MARGIN, TOP_FRACTION)

NOTA_INTERPRETAZIONE = (
    "Le mappe evidenziano le regioni che aumentano la probabilita' di melanoma "
    "stimata dal modello. Per un esito benigno indicano le aree giudicate piu' "
    "sospette, senza che la soglia sia stata superata. La mappa indica dove il "
    "modello ha concentrato l'attenzione, non quali strutture abbia riconosciuto."
)

_DIREZIONI = ["destra", "inferiore destra", "inferiore", "inferiore sinistra",
              "sinistra", "superiore sinistra", "superiore", "superiore destra"]


# ----------------------------------------------------------------------------
# Segmentazione euristica della lesione
# ----------------------------------------------------------------------------
def _field_of_view(L):
  """Esclude la vignettatura nera del dermatoscopio (aree molto scure al bordo)."""
  dark = (L < 25).astype(np.uint8)
  n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
  h, w = L.shape
  fov = np.ones((h, w), bool)
  for i in range(1, n):
    x, y, bw, bh, area = stats[i]
    touches = x == 0 or y == 0 or x + bw == w or y + bh == h
    if touches and area > 0.002 * h * w:
      fov[lab == i] = False
  return fov


def segment_lesion(rgb):
  """Otsu sul canale L* + pulizia morfologica + componente piu' centrale.

  Restituisce (maschera bool, info). info["affidabile"] e' False se l'area e'
  implausibile o la maschera tocca troppo i bordi dell'immagine.
  """
  lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
  L = lab[..., 0]
  h, w = L.shape
  info = {"metodo": "soglia di Otsu sulla luminosita' (euristica)",
          "affidabile": False}
  fov = _field_of_view(L)
  blur = cv2.GaussianBlur(L, (7, 7), 0)
  vals = blur[fov].reshape(-1, 1)
  if vals.size < 100:
    return np.zeros((h, w), bool), info

  t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  cand = ((blur <= t) & fov).astype(np.uint8)
  cand = cv2.morphologyEx(
      cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
  cand = cv2.morphologyEx(
      cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

  n, lab_cc, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
  best, best_score = 0, 0.0
  half_diag = np.hypot(h, w) / 2
  for i in range(1, n):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 0.005 * h * w:
      continue
    x, y, bw, bh = stats[i, :4]
    touches = x == 0 or y == 0 or x + bw == w or y + bh == h
    d = np.hypot(cents[i][0] - w / 2, cents[i][1] - h / 2) / half_diag
    score = area * (1 - 0.7 * d) * (0.5 if touches else 1.0)
    if score > best_score:
      best, best_score = i, score
  if best == 0:
    return np.zeros((h, w), bool), info

  comp = (lab_cc == best).astype(np.uint8)
  contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  mask = np.zeros_like(comp)
  cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)  # riempie i buchi
  mask = mask.astype(bool)

  area_frac = float(mask.mean())
  border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
  border_frac = float(border.mean())
  info.update({
      "area_frazione": round(area_frac, 3),
      "bordo_immagine_frazione": round(border_frac, 3),
      "affidabile": bool(0.02 <= area_frac <= 0.85 and border_frac <= 0.25),
  })
  return mask, info


# ----------------------------------------------------------------------------
# Utilita'
# ----------------------------------------------------------------------------
def _lab_float(rgb):
  """CIELAB in scala standard: L* in [0,100], a*/b* centrati in 0."""
  lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
  lab[..., 0] *= 100.0 / 255.0
  lab[..., 1:] -= 128.0
  return lab


def _direction(dx, dy):
  """Direzione testuale; dy positivo verso il basso (coordinate immagine)."""
  ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
  return _DIREZIONI[int(((ang + 22.5) % 360.0) // 45)]


def _centroid(mask):
  m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
  if m["m00"] == 0:
    return None
  return m["m10"] / m["m00"], m["m01"] / m["m00"], m


def _pct(x):
  return None if x is None else int(round(100.0 * float(x)))


# ----------------------------------------------------------------------------
# Descrittori della lesione (forma e colore)
# ----------------------------------------------------------------------------
def _asymmetry(mask):
  """Asimmetria rispetto ai due assi principali: XOR(maschera, ribaltata)/2A."""
  c = _centroid(mask)
  if c is None:
    return None
  cx, cy, m = c
  theta = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
  size = 2 * max(mask.shape)
  R = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
  R[0, 2] += size / 2 - cx
  R[1, 2] += size / 2 - cy
  al = cv2.warpAffine(mask.astype(np.uint8), R, (size, size),
                      flags=cv2.INTER_NEAREST).astype(bool)
  area = al.sum()
  if area == 0:
    return None
  a1 = np.logical_xor(al, al[::-1, :]).sum() / (2.0 * area)
  a2 = np.logical_xor(al, al[:, ::-1]).sum() / (2.0 * area)
  return float((a1 + a2) / 2.0)


def _solidity(mask):
  contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    return None
  cnt = max(contours, key=cv2.contourArea)
  hull_area = cv2.contourArea(cv2.convexHull(cnt))
  return float(mask.sum() / hull_area) if hull_area > 0 else None


def _color_clusters(lab, mask, k=6, merge_de=12.0, min_share=0.05):
  """Numero di toni distinti nella lesione (k-means in CIELAB + fusione per dE)."""
  px = lab[mask].astype(np.float32)
  if len(px) < 60:
    return None, False
  rng = np.random.default_rng(0)
  if len(px) > 5000:
    px = px[rng.choice(len(px), 5000, replace=False)]
  k = min(k, len(px))
  crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
  cv2.setRNGSeed(0)
  _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
  shares = np.bincount(labels.ravel(), minlength=k) / len(px)
  groups = []  # [centro, quota]
  for i in np.argsort(-shares):
    c, s = centers[i].astype(np.float64), float(shares[i])
    for g in groups:
      if np.linalg.norm(g[0] - c) < merge_de:
        g[0] = (g[0] * g[1] + c * s) / (g[1] + s)
        g[1] += s
        break
    else:
      groups.append([c, s])
  significant = [g for g in groups if g[1] >= min_share]
  bluish = any(g[0][2] < 0.0 for g in significant)  # b* negativo
  return len(significant), bool(bluish)


def lesion_descriptors(rgb, mask, seg_info):
  lab = _lab_float(rgb)
  d = {"segmentazione_affidabile": seg_info.get("affidabile", False)}
  if not mask.any():
    d["nota"] = "lesione non individuata dalla segmentazione automatica"
    return d
  area = int(mask.sum())
  asym = _asymmetry(mask)
  sol = _solidity(mask)
  n_col, bluish = _color_clusters(lab, mask)
  d.update({
      "area_pct_immagine": _pct(area / mask.size),
      "diametro_equivalente_px": int(round(2 * np.sqrt(area / np.pi))),
      "indice_asimmetria": None if asym is None else round(asym, 2),
      "asimmetria": None if asym is None else (
          "bassa" if asym < 0.10 else "moderata" if asym < 0.25 else "marcata"),
      "solidita_contorno": None if sol is None else round(sol, 2),
      "regolarita_contorno": None if sol is None else (
          "regolare" if sol >= 0.95 else
          "moderatamente irregolare" if sol >= 0.85 else "irregolare"),
      "toni_colore_distinti": n_col,
      "variegatura_colore": None if n_col is None else (
          "omogenea" if n_col <= 2 else "moderata" if n_col == 3 else "marcata"),
      "presenza_toni_bluastri_grigiastri": bluish,
  })
  return d


# ----------------------------------------------------------------------------
# Descrittori della mappa Grad-CAM++
# ----------------------------------------------------------------------------
def cam_descriptors(cam, mask, lab):
  h, w = cam.shape
  cam = np.clip(cam.astype(np.float32), 0, 1)
  total = float(cam.sum()) + 1e-8
  hot = cam >= CAM_HOT_THRESHOLD
  py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)

  n_cc, _, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), 8)
  n_regions = int(sum(stats[i, cv2.CC_STAT_AREA] >= 0.005 * h * w
                      for i in range(1, n_cc)))
  edge_dist = min(px, py, w - 1 - px, h - 1 - py)

  d = {
      "regioni_calde": n_regions,
      "area_calda_pct_immagine": _pct(hot.mean()),
      "picco_vicino_bordo_immagine": bool(edge_dist < 0.06 * w),
  }
  if not mask.any():
    d.update({"energia_dentro_lesione_pct": None, "zona_picco": None,
              "direzione_picco": None})
    return d

  din = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
  dout = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
  rmax = float(din.max()) + 1e-8

  # Posizione del picco
  if mask[py, px]:
    rel = din[py, px] / rmax
    zona = ("centro della lesione" if rel > 0.6 else
            "zona intermedia della lesione" if rel > 0.25 else
            "margine della lesione")
  else:
    zona = ("cute perilesionale" if dout[py, px] <= 0.06 * w else
            "cute distante dalla lesione")
  cx, cy, _ = _centroid(mask)
  r_eq = np.sqrt(mask.sum() / np.pi)
  dx, dy = px - cx, py - cy
  direzione = "centrale" if np.hypot(dx, dy) < 0.2 * r_eq else _direction(dx, dy)

  # Estensione
  copertura = (hot & mask).sum() / mask.sum()
  fuori = (hot & ~mask).sum() / max(int(hot.sum()), 1)
  if n_regions >= 3:
    distribuzione = "frammentata in piu' regioni"
  elif copertura >= 0.5:
    distribuzione = "estesa su gran parte della lesione"
  else:
    distribuzione = "focale"

  # Margini vs centro della lesione
  band = mask & (din <= 0.25 * rmax)
  core = mask & (din > 0.6 * rmax)
  if band.sum() >= 30 and core.sum() >= 30:
    ratio = cam[band].mean() / (cam[core].mean() + 1e-6)
    margine_centro = ("prevalente sui margini" if ratio > 1.25 else
                      "prevalente al centro" if ratio < 0.8 else
                      "distribuita tra centro e margini")
  else:
    margine_centro = None

  # Colore della regione calda rispetto al resto della lesione
  hot_in, rest = hot & mask, mask & ~hot
  colore, deltas = [], None
  if hot_in.sum() >= 30 and rest.sum() >= 30:
    dL = float(lab[hot_in, 0].mean() - lab[rest, 0].mean())
    da = float(lab[hot_in, 1].mean() - lab[rest, 1].mean())
    db = float(lab[hot_in, 2].mean() - lab[rest, 2].mean())
    deltas = {"delta_L": round(dL, 1), "delta_a": round(da, 1),
              "delta_b": round(db, 1)}
    if dL < -5:
      colore.append("piu' scura")
    elif dL > 5:
      colore.append("piu' chiara")
    if db < -4:
      colore.append("tonalita' piu' bluastra/grigiastra")
    if da > 4:
      colore.append("tonalita' piu' rossastra")
    if not colore:
      colore.append("colore simile al resto della lesione")

  d.update({
      "energia_dentro_lesione_pct": _pct((cam * mask).sum() / total),
      "zona_picco": zona,
      "direzione_picco": direzione,
      "copertura_lesione_pct": _pct(copertura),
      "area_calda_fuori_lesione_pct": _pct(fuori),
      "distribuzione": distribuzione,
      "attivazione_margini_vs_centro": margine_centro,
      "regione_calda_rispetto_al_resto": colore or None,
      "differenze_colore_CIELAB": deltas,
  })
  return d


def compare_cams(c1, c2):
  t1 = np.quantile(c1, 1 - TOP_FRACTION)
  t2 = np.quantile(c2, 1 - TOP_FRACTION)
  a, b = c1 >= t1, c2 >= t2
  iou = (a & b).sum() / max(int((a | b).sum()), 1)
  p1 = np.unravel_index(int(np.argmax(c1)), c1.shape)
  p2 = np.unravel_index(int(np.argmax(c2)), c2.shape)
  dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1]) / np.hypot(*c1.shape)
  corr = float(np.corrcoef(c1.ravel(), c2.ravel())[0, 1])
  return {
      "iou_regioni_top20": round(float(iou), 2),
      "sovrapposizione": ("ampia" if iou > 0.5 else
                          "parziale" if iou >= 0.25 else "scarsa"),
      "distanza_picchi_pct_diagonale": _pct(dist),
      "correlazione_mappe": None if np.isnan(corr) else round(corr, 2),
  }


def prediction_summary(p, tau):
  margin = p - tau
  if abs(margin) < NEAR_THRESHOLD_MARGIN:
    sicurezza = "bassa, vicina alla soglia"
  elif p >= 0.9 or p <= 0.1:
    sicurezza = "elevata"
  else:
    sicurezza = "moderata"
  return {"probabilita_melanoma": round(float(p), 3),
          "esito": "Melanoma" if p >= tau else "Benigno",
          "sicurezza": sicurezza}


# ----------------------------------------------------------------------------
# Caso completo e avvertenze a regole
# ----------------------------------------------------------------------------
def compute_case_features(rgb, mask, seg_info, cams, probs, tau, dataset_name):
  """Dizionario inviato all'LLM. NON contiene la ground truth."""
  lab = _lab_float(rgb)
  features = {
      "dataset": dataset_name,
      "soglia_decisionale": tau,
      "nota_interpretazione": NOTA_INTERPRETAZIONE,
      "lesione": lesion_descriptors(rgb, mask, seg_info),
      "modelli": {},
  }
  for k in MODEL_KEYS:
    features["modelli"][k] = {
        "nome": MODEL_NAMES[k],
        "predizione": prediction_summary(probs[k], tau),
        "mappa": cam_descriptors(cams[k], mask, lab),
    }
  cmp = compare_cams(cams["resnet50"], cams["mobilevit_s"])
  e1 = features["modelli"]["resnet50"]["predizione"]["esito"]
  e2 = features["modelli"]["mobilevit_s"]["predizione"]["esito"]
  cmp["stesso_esito"] = e1 == e2
  features["confronto_modelli"] = cmp
  return features


def rule_warnings(features):
  """Avvertenze deterministiche: non dipendono dall'LLM e vengono verificate."""
  out = []
  seg_ok = features["lesione"].get("segmentazione_affidabile", False)
  if not seg_ok:
    out.append({"codice": "SEGMENTAZIONE_INCERTA", "modello": None,
                "testo": "La delimitazione automatica della lesione e' incerta: "
                         "i descrittori di posizione e colore vanno letti con cautela."})
  for k in MODEL_KEYS:
    m = features["modelli"][k]
    name, mp, pr = m["nome"], m["mappa"], m["predizione"]
    en = mp.get("energia_dentro_lesione_pct")
    if en is not None and en < 100 * MIN_LESION_ENERGY:
      out.append({"codice": "FUORI_LESIONE", "modello": k,
                  "testo": f"{name}: solo il {en}% dell'attivazione cade sulla "
                           "lesione; la predizione potrebbe basarsi su elementi "
                           "non clinicamente rilevanti."})
    elif mp.get("zona_picco") in ("cute perilesionale", "cute distante dalla lesione"):
      out.append({"codice": "PICCO_FUORI_LESIONE", "modello": k,
                  "testo": f"{name}: il punto di massima attivazione si trova "
                           f"sulla {mp['zona_picco']}."})
    if mp.get("picco_vicino_bordo_immagine"):
      out.append({"codice": "MARGINE_IMMAGINE", "modello": k,
                  "testo": f"{name}: il picco di attivazione e' vicino al bordo "
                           "dell'immagine, dove possono trovarsi artefatti."})
    if pr["sicurezza"].startswith("bassa"):
      out.append({"codice": "VICINO_SOGLIA", "modello": k,
                  "testo": f"{name}: probabilita' {pr['probabilita_melanoma']} "
                           f"vicina alla soglia {features['soglia_decisionale']}; "
                           "esito poco stabile."})
  if not features["confronto_modelli"]["stesso_esito"]:
    out.append({"codice": "DISACCORDO", "modello": None,
                "testo": "I due modelli producono esiti diversi: il caso merita "
                         "una valutazione specialistica attenta."})
  return out
