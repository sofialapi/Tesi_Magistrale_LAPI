"""Schede di spiegazione per il clinico sui modelli multimodali ISIC 2024.

Per ogni lesione: crop TBP, Grad-CAM++ del ramo visivo, percorso della decisione
(riferimento -> sola immagine -> immagine + dati clinici), contributi SHAP dei dati
clinici e spiegazione testuale (GPT-OSS-20B via Groq, come nella Sezione 6.2).

Richiede i risultati di shap_run (outputs/shap/shap_values_<case>.csv).
Uso, dalla radice del progetto:

  # due casi scelti automaticamente: vero positivo e decisione ribaltata dai dati clinici
  python -m XAI_SHAP.shap_card

  # lesioni specifiche, altro modello, testo da template (senza API)
  python -m XAI_SHAP.shap_card --ids ISIC_4676617 --case hybrid_multimodal --no_llm

  # altre tipologie: vero_positivo, ribaltata_da_clinica, ribaltata_in_negativo,
  #                  falso_negativo, falso_positivo
  python -m XAI_SHAP.shap_card --kinds falso_negativo falso_positivo
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import pandas as pd

from XAI_SHAP import shap_config as cfg
from XAI_SHAP.card_text import (LLM_MODEL, SMALL_AREA_MM2,
                                GroqCardExplainer, TemplateCardExplainer, case_features,
                                rule_warnings, verify)

RAW_DIR = os.path.join(cfg.ROOT, "data", "raw", "ISIC")
CARD_FIG_DIR = os.path.join(cfg.FIG_DIR, "schede")
CARD_OUT_DIR = os.path.join(cfg.OUT_DIR, "schede")
MODEL_LABEL = {"cnn_multimodal": "Caso 3: ResNet-50 + dati clinici",
               "hybrid_multimodal": "Caso 4: MobileViT-S + dati clinici"}


def parse_args():
  ap = argparse.ArgumentParser(description="Schede SHAP + Grad-CAM++ per il clinico")
  ap.add_argument("--case", default="cnn_multimodal", choices=list(cfg.CASES))
  ap.add_argument("--ids", nargs="+")
  ap.add_argument("--kinds", nargs="+", default=["vero_positivo", "ribaltata_da_clinica"])
  ap.add_argument("--allow_flagged", action="store_true",
                  help="nella selezione automatica ammette dati mancanti e aree < 2 mm²")
  ap.add_argument("--no_llm", action="store_true")
  ap.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="medium")
  ap.add_argument("--temperature", type=float, default=0.3)
  ap.add_argument("--dpi", type=int, default=300)
  return ap.parse_args()


# ----------------------------------------------------------------------------
# Selezione dei casi
# ----------------------------------------------------------------------------
def _flipped_up(mal, delta):
  c = mal[(mal.base_tab < 0) & (mal.logit >= 0)]
  firm = c[c.prob >= 0.65]
  c = firm if len(firm) else c
  return delta[c.index].idxmax()


def select_cases(d, kinds, allow_flagged):
  """Restituisce [(tipologia, isic_id)] scegliendo, per tipologia, il caso piu' netto."""
  pool = d
  if not allow_flagged:
    flagged = d[[c for c in d.columns if c.startswith("val_")]].isna().any(axis=1)
    pool = d[~flagged & (d["val_tbp_lv_areaMM2"] > SMALL_AREA_MM2)]
  mal, ben = pool[pool.target == 1], pool[pool.target == 0]
  delta = pool["logit"] - pool["base_tab"]
  rules = {
      # maligna riconosciuta gia' dall'immagine, in cui i dati clinici rafforzano di piu'
      "vero_positivo": lambda: delta[mal.index][(mal.base_tab >= 0) & (mal.logit >= 0)].idxmax(),
      # maligna mancata dalla sola immagine e recuperata dai dati clinici
      # (tra quelle con stima finale >= 0.65, se esistono, per un esempio non al limite)
      "ribaltata_da_clinica": lambda: _flipped_up(mal, delta),
      # maligna riconosciuta dall'immagine ma persa per i dati clinici
      "ribaltata_in_negativo": lambda: delta[mal.index][(mal.base_tab >= 0) & (mal.logit < 0)].idxmin(),
      "falso_negativo": lambda: mal.logit.idxmin(),
      "falso_positivo": lambda: ben.logit.idxmax(),
  }
  out = []
  for k in kinds:
    try:
      out.append((k, pool.loc[rules[k](), "isic_id"]))
    except (ValueError, KeyError):
      print(f"[AVVISO] nessun caso per la tipologia '{k}'")
  return out


# ----------------------------------------------------------------------------
# Modello e Grad-CAM++
# ----------------------------------------------------------------------------
class _ImageOnly:
  """Il modello multimodale visto come funzione della sola immagine (clinica fissata)."""

  def __new__(cls, model, clin):
    import torch.nn as nn

    class Wrapper(nn.Module):
      def __init__(self):
        super().__init__()
        self.m = model
        self.register_buffer("clin", clin)

      def forward(self, img):
        return self.m(img, self.clin.expand(img.shape[0], -1))
    return Wrapper()


class _LogOddsTarget:
  """Bersaglio della mappa: log-odds di malignita' z1 - z0 (stessa grandezza di SHAP)."""

  def __call__(self, out):
    return out[..., 1] - out[..., 0]


def target_layer(model, backbone):
  vb = model.visual_backbone.model
  if backbone == "cnn":
    return vb.layer4[-1].conv3, "visual_backbone.model.layer4[-1].conv3"
  last = vb.stages[-1]
  if hasattr(last, "blocks") and len(last.blocks) > 0:
    return last.blocks[-1], "visual_backbone.model.stages[-1].blocks[-1]"
  return last, "visual_backbone.model.stages[-1]"


def lesion_attention(cam, rgb):
  """Quota della mappa sulla lesione (segmentazione euristica della Sez. 6.2)."""
  try:
    from XAI_NL.nl_cam_features import segment_lesion
    mask, info = segment_lesion(rgb)
  except Exception as e:  # noqa: BLE001
    print(f"  segmentazione non disponibile ({e})")
    return None, {"quota_lesione_pct": None, "segmentazione_affidabile": None}
  reliable = bool(info.get("affidabile")) if isinstance(info, dict) else False
  if not reliable or mask is None or not mask.any():
    return mask, {"quota_lesione_pct": None, "segmentazione_affidabile": reliable}
  q = int(round(100 * float((cam * mask).sum()) / max(float(cam.sum()), 1e-8)))
  return mask, {"quota_lesione_pct": q, "segmentazione_affidabile": True}


def load_rgb(path, size=None):
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
  return img if size is None else cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
  args = parse_args()
  import torch
  from pytorch_grad_cam import GradCAMPlusPlus

  from XAI_SHAP.shap_data import DermalMultimodalDataset, fit_processor, iter_folds, load_metadata, load_model

  case = args.case
  shap_csv = os.path.join(cfg.OUT_DIR, f"shap_values_{case}.csv")
  if not os.path.exists(shap_csv):
    raise SystemExit(f"{shap_csv} assente: esegui prima python -m XAI_SHAP.shap_run")
  d = pd.read_csv(shap_csv)
  if args.ids:
    chosen = [("scelta_manuale", i) for i in args.ids]
  else:
    chosen = select_cases(d, args.kinds, args.allow_flagged)
  if not chosen:
    raise SystemExit("Nessun caso selezionato.")

  os.makedirs(CARD_FIG_DIR, exist_ok=True)
  os.makedirs(CARD_OUT_DIR, exist_ok=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  meta = load_metadata()
  folds = {k: (tr, va) for k, tr, va in iter_folds(meta)}
  explainer = TemplateCardExplainer() if args.no_llm else GroqCardExplainer(
      model=LLM_MODEL, reasoning_effort=args.reasoning_effort, temperature=args.temperature,
      cache_dir=os.path.join(CARD_OUT_DIR, "cache"))
  backbone = cfg.CASES[case]["backbone"]
  models = {}
  summary = []

  for n, (kind, isic) in enumerate(chosen, 1):
    rows = d[d.isic_id == isic]
    if rows.empty:
      print(f"[AVVISO] {isic} non presente in {shap_csv}")
      continue
    row = rows.iloc[0]
    fold = int(row["fold"])
    tr, va = folds[fold]
    if fold not in models:
      proc, X_tr, _ = fit_processor(meta.iloc[tr].reset_index(drop=True))
      models[fold] = (proc, load_model(case, fold, X_tr.shape[1], device))
    proc, model = models[fold]
    mrow = meta[meta.isic_id == isic]
    clin = proc.transform(mrow.reset_index(drop=True)).astype(np.float32)

    ds = DermalMultimodalDataset(image_ids=np.array([isic]), labels=mrow["target"].values,
                                 image_dir=cfg.IMAGE_DIR, clinical_matrix=clin,
                                 mode="multimodal", is_training=False)
    img_t, clin_t, _ = ds[0]
    img_t, clin_t = img_t.unsqueeze(0).to(device), clin_t.unsqueeze(0).to(device)

    with torch.no_grad():
      z = model(img_t, clin_t)[0]
    logit_model = float(z[1] - z[0])
    if abs(logit_model - row["logit"]) > 1e-3:
      print(f"[AVVISO] {isic}: logit del modello {logit_model:.4f} diverso da quello del "
            f"CSV SHAP {row['logit']:.4f}")

    wrapper = _ImageOnly(model, clin_t).to(device).eval()
    layer, layer_name = target_layer(model, backbone)
    cam_fn = GradCAMPlusPlus(model=wrapper, target_layers=[layer])
    cam = cam_fn(input_tensor=img_t, targets=[_LogOddsTarget()])[0]
    if hasattr(cam_fn, "activations_and_grads"):
      cam_fn.activations_and_grads.release()

    h, w = cam.shape
    model_rgb = load_rgb(os.path.join(cfg.IMAGE_DIR, f"{isic}.jpg"), (w, h))
    raw_path = os.path.join(RAW_DIR, f"{isic}.jpg")
    crop_rgb = load_rgb(raw_path) if os.path.exists(raw_path) else model_rgb
    mask, cam_info = lesion_attention(cam, model_rgb)

    features = case_features(row, cfg.VAR_ORDER, MODEL_LABEL[case], cam_info)
    warns = rule_warnings(features)
    features["avvertenze_calcolate"] = [x["testo"] for x in warns]
    try:
      expl = explainer.explain(features)
    except Exception as e:  # noqa: BLE001
      print(f"[{isic}] LLM non disponibile ({e}); uso il template.")
      expl = TemplateCardExplainer().explain(features)
    sec = expl["sezioni"]
    ver = verify(sec, features, warns)
    extra = [x["testo"] for x in warns if x["codice"] in ver["avvertenze_mancanti"]]

    truth = "Maligna" if int(row["target"]) == 1 else "Benigna"
    src = "GPT-OSS-20B (Groq)" if expl["meta"]["explainer"] == "llm" else "template a regole"
    footer = (
        f"Spiegazione generata automaticamente ({src}) a partire da descrittori quantitativi; "
        "non costituisce una diagnosi. Percorso della decisione: probabilità per una lesione "
        f"di riferimento (media su {cfg.BACKGROUND_SIZE} lesioni del fold di addestramento), con "
        "la sola immagine (dati clinici sostituiti dai valori di riferimento) e con immagine e "
        "dati clinici. Contributi dei dati clinici: valori SHAP esatti sul log-odds di "
        "malignità (Sezione 6.3), che descrivono quanto il modello ha usato ciascun dato e non "
        "relazioni causali. Grad-CAM++ calcolata sul log-odds con dati clinici fissati: la "
        "griglia dell'ultimo strato è 7×7 su un crop di 224×224, quindi la mappa indica solo "
        "la zona approssimativa; contorno tratteggiato: segmentazione automatica. "
        f"Checkpoint del fold {fold}, lesione di validazione; la verità non è fornita al "
        "generatore di testo.")
    stem = f"scheda_{case}_{n:02d}_{isic}_{kind}"
    fig_path = os.path.join(CARD_FIG_DIR, stem + ".png")
    from XAI_SHAP.card_figure import render_card
    render_card(fig_path, crop_rgb, model_rgb, cam, mask,
                {"image_id": isic, "verita": truth, "colore_modello": cfg.CASE_COLORS[case]},
                features, sec, extra_warnings=extra, footer=footer, dpi=args.dpi)

    record = {"isic_id": isic, "tipologia": kind, "verita": truth, "caso": case, "fold": fold,
              "logit_modello": logit_model, "logit_csv": float(row["logit"]),
              "target_layer": layer_name, "descrittori": features,
              "avvertenze_regole": warns, "spiegazione": expl, "avvertenze_aggiunte": extra,
              "verifica": ver, "figura": fig_path}
    with open(os.path.join(CARD_OUT_DIR, stem + ".json"), "w", encoding="utf-8") as f:
      json.dump(record, f, ensure_ascii=False, indent=1, default=float)
    summary.append({"isic_id": isic, "tipologia": kind, "verita": truth, "fold": fold,
                    "p_finale": features["probabilita"]["immagine_e_dati_clinici"],
                    "p_solo_immagine": features["probabilita"]["solo_immagine"],
                    "ruolo_dati_clinici": features["ruolo_dati_clinici"],
                    "quota_attenzione_lesione_pct": cam_info["quota_lesione_pct"],
                    "avvertenze": ";".join(x["codice"] for x in warns),
                    "explainer": expl["meta"]["explainer"], "verifica_superata": ver["superata"],
                    "numeri_non_verificati": " ".join(f"{x:g}" for x in ver["numeri_non_verificati"]),
                    "avvertenze_mancanti": ";".join(ver["avvertenze_mancanti"]),
                    "figura": fig_path})
    print(f"[{n}/{len(chosen)}] {isic} ({kind}, {truth}, fold {fold}) -> P {features['probabilita']['immagine_e_dati_clinici']:.3f} "
          f"(sola immagine {features['probabilita']['solo_immagine']:.3f}) | verifica "
          f"{'OK' if ver['superata'] else 'NON superata'} | {os.path.relpath(fig_path, cfg.ROOT)}")

  if summary:
    path = os.path.join(CARD_OUT_DIR, f"schede_{case}_riepilogo.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
      wr = csv.DictWriter(f, fieldnames=list(summary[0]))
      wr.writeheader()
      wr.writerows(summary)
    print(f"Riepilogo: {os.path.relpath(path, cfg.ROOT)}")


if __name__ == "__main__":
  main()
