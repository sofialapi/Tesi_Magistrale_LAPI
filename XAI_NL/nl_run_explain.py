"""Spiegazioni in linguaggio naturale delle mappe Grad-CAM++ (BCN20000 e HAM10000).

Da lanciare dalla root del progetto (~/Tesi_Magistrale_LAPI).

  # 3 melanomi + 3 benigni dal fold di validazione di default (BCN: 1, HAM: 4)
  python -m XAI_NL.nl_run_explain --dataset bcn --num_samples 6
  python -m XAI_NL.nl_run_explain --dataset ham --num_samples 6

  # ID specifici: ogni immagine viene spiegata dai checkpoint del fold in cui
  # e' di validazione (se disponibili)
  python -m XAI_NL.nl_run_explain --dataset ham --ids ISIC_0028086

  # forzare un fold (gli ID fuori da quel fold vengono marcati "training")
  python -m XAI_NL.nl_run_explain --dataset bcn --ids ISIC_0000002 --fold 1

  # stessi casi con i modelli addestrati dagli script raw (Sezione 6.1.3)
  python -m XAI_NL.nl_run_explain --dataset ham_raw --ids ISIC_0028086

  # senza API (testo da template)
  python -m XAI_NL.nl_run_explain --dataset ham --num_samples 6 --no_llm
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .nl_cam_features import compute_case_features, rule_warnings, segment_lesion
from .nl_config import (DATASETS, FEATURE_SIZE, FIGURE_DIR, LLM_MODEL, MODEL_KEYS,
                        OUTPUT_DIR, available_folds, resolve_checkpoint)
from .nl_explainer import GroqExplainer, TemplateExplainer
from .nl_figure import render_case_figure
from .nl_verify import verify_explanation

ID_COLS = ("image_id", "image_name", "isic_id", "image")
TARGET_COLS = ("target", "MEL", "label")


def parse_args():
  ap = argparse.ArgumentParser(description="Grad-CAM++ + spiegazione NL (GPT-OSS-20B)")
  ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
  ap.add_argument("--ids", nargs="+", help="ID specifici (es. ISIC_0028086)")
  ap.add_argument("--fold", type=int, help="fold dei checkpoint (default da config)")
  ap.add_argument("--num_samples", type=int, default=6)
  ap.add_argument("--selection", choices=["random", "first"], default="random")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--threshold", type=float, default=0.5)
  ap.add_argument("--overlay_on", choices=["input", "display"], default="input",
                  help="mappe sovrapposte all'input del modello o all'immagine mostrata")
  ap.add_argument("--no_llm", action="store_true", help="usa il template a regole")
  ap.add_argument("--llm_model", default=LLM_MODEL)
  ap.add_argument("--reasoning_effort", choices=["low", "medium", "high"],
                  default="medium")
  ap.add_argument("--temperature", type=float, default=0.3)
  ap.add_argument("--no_contour", action="store_true")
  ap.add_argument("--figure_dir")
  ap.add_argument("--output_dir")
  ap.add_argument("--dpi", type=int, default=300)
  return ap.parse_args()


# ----------------------------------------------------------------------------
# Dati e fold
# ----------------------------------------------------------------------------
def normalize_id(x):
  s = os.path.basename(str(x))
  if s.lower().endswith((".jpg", ".jpeg", ".png")):
    s = os.path.splitext(s)[0]
  return s.replace("_downsampled", "")


def build_index(root):
  idx = {}
  if not root or not os.path.isdir(root):
    return idx
  for f in os.listdir(root):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
      idx.setdefault(normalize_id(f), os.path.join(root, f))
  return idx


def detect_columns(df):
  id_col = next((c for c in ID_COLS if c in df.columns), None)
  if id_col is None:
    raise KeyError(f"Nessuna colonna ID tra {ID_COLS}: {list(df.columns)}")
  tgt = next((c for c in TARGET_COLS if c in df.columns), None)
  if tgt is None and "dx" in df.columns:  # HAM10000_metadata.csv
    df["target"] = (df["dx"] == "mel").astype(int)
    tgt = "target"
  if tgt is None:
    raise KeyError(f"Nessuna colonna target tra {TARGET_COLS} o 'dx'")
  return id_col, tgt


def rebuild_kfold(df, img_dir, seed, id_col, tgt, tag):
  """Replica gli script di training: mappatura file, dropna, StratifiedKFold."""
  from sklearn.model_selection import StratifiedKFold
  existing = {f: os.path.join(img_dir, f) for f in os.listdir(img_dir)
              if f.endswith(".jpg")}
  paths = []
  for name in df[id_col]:
    base = str(name).replace(".jpg", "")
    cands = [f"{base}.jpg", f"{base}_downsampled.jpg",
             base.replace("_downsampled", "") + ".jpg"]
    paths.append(next((existing[c] for c in cands if c in existing), None))
  df = df.copy()
  df["filepath"] = paths
  initial = len(df)
  df = df.dropna(subset=["filepath"]).reset_index(drop=True)
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
  df["fold"] = 0
  for k, (_, val_idx) in enumerate(skf.split(df, df[tgt]), start=1):
    df.loc[val_idx, "fold"] = k
  print(f"[{tag}] fold ricostruiti su {len(df)} di {initial} righe del CSV "
        "(deve coincidere con il numero di campioni stampato dal training).")
  print(f"[{tag}] campioni di validazione per fold: "
        + ", ".join(f"{k}={int((df['fold'] == k).sum())}" for k in range(1, 6)))
  return df


def load_dataframe(cfg, seed):
  df = pd.read_csv(cfg["csv"])
  id_col, tgt = detect_columns(df)
  if cfg["folds"] == "stratified_kfold":
    df = rebuild_kfold(df, cfg["input_dir"], cfg.get("kfold_seed", seed),
                       id_col, tgt, cfg["name"])
  elif "fold" not in df.columns:
    raise KeyError("Colonna 'fold' assente nel CSV")
  df["_id"] = df[id_col].map(normalize_id)
  df["_target"] = df[tgt].astype(int)
  df["fold"] = df["fold"].astype(int)
  return df


def select_samples(df, args, fold, input_idx):
  df = df[df["_id"].isin(input_idx)]
  if args.ids:
    wanted = [normalize_id(i) for i in args.ids]
    sel = df[df["_id"].isin(wanted)].drop_duplicates("_id")
    missing = sorted(set(wanted) - set(sel["_id"]))
    if missing:
      print(f"[AVVISO] ID non trovati nel CSV o tra le immagini: {missing}")
    return sel
  df = df[df["fold"] == fold]
  n_mal = args.num_samples // 2
  mal, ben = df[df["_target"] == 1], df[df["_target"] == 0]
  if args.selection == "random":
    mal = mal.sample(min(n_mal, len(mal)), random_state=args.seed)
    ben = ben.sample(min(args.num_samples - len(mal), len(ben)),
                     random_state=args.seed)
  else:
    mal = mal.head(n_mal)
    ben = ben.head(args.num_samples - len(mal))
  return pd.concat([mal, ben])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
  args = parse_args()
  cfg = DATASETS[args.dataset]
  fig_dir = args.figure_dir or os.path.join(FIGURE_DIR, args.dataset)
  out_dir = args.output_dir or os.path.join(OUTPUT_DIR, args.dataset)
  os.makedirs(fig_dir, exist_ok=True)
  os.makedirs(out_dir, exist_ok=True)

  input_idx = build_index(cfg["input_dir"])
  display_idx = build_index(cfg["display_dir"])
  print(f"Immagini indicizzate: {len(input_idx)} input, {len(display_idx)} display")
  if not input_idx:
    raise SystemExit(f"Nessuna immagine in {cfg['input_dir']}")

  df = load_dataframe(cfg, args.seed)
  forced_fold = args.fold
  default_fold = args.fold or cfg["default_fold"]
  samples = select_samples(df, args, default_fold, input_idx)
  if samples.empty:
    raise SystemExit("Nessun caso selezionato.")

  # import qui: torch serve solo per l'esecuzione completa
  import torch
  from .nl_models import CamRunner, load_rgb, to_tensor

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  runners_cache = {}

  def get_runners(fold):
    if fold not in runners_cache:
      ck = {"resnet50": resolve_checkpoint(cfg["cnn_ckpt"], fold),
            "mobilevit_s": resolve_checkpoint(cfg["hybrid_ckpt"], fold)}
      if any(p is None or not os.path.exists(p) for p in ck.values()):
        runners_cache[fold] = None
      else:
        runners_cache[fold] = {
            "resnet50": CamRunner("resnet50", ck["resnet50"], device),
            "mobilevit_s": CamRunner("mobilevit_s", ck["mobilevit_s"], device),
            "ckpt": ck}
        for k in MODEL_KEYS:
          print(f"[fold {fold}] {k}: {ck[k]} | target layer "
                f"{runners_cache[fold][k].target_layer_name}")
    return runners_cache[fold]

  if args.no_llm:
    explainer = TemplateExplainer()
  else:
    explainer = GroqExplainer(
        model=args.llm_model, reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        cache_dir=os.path.join(OUTPUT_DIR, "cache"))
  print(f"Casi selezionati: {len(samples)} | tau={args.threshold} | "
        f"explainer={explainer.name}")

  summary = []
  for n, (_, row) in enumerate(samples.iterrows(), start=1):
    img_id, img_fold = row["_id"], int(row["fold"])
    gt = "Melanoma" if row["_target"] == 1 else "Benigno"
    ck_fold = forced_fold if forced_fold is not None else (
        img_fold if args.ids else default_fold)
    runners = get_runners(ck_fold)
    if runners is None:
      print(f"[{img_id}] e' nel fold {img_fold}, ma mancano i checkpoint di quel "
            f"fold (ResNet-50 disponibile per i fold "
            f"{available_folds(cfg['cnn_ckpt'])}). Usa --fold <k> per forzare: "
            "l'immagine verra' marcata come training. Caso saltato.")
      continue
    split = "validazione" if img_fold == ck_fold else "training"

    # input identico al val_tf del training (risoluzione nativa se input_size None)
    inp = load_rgb(input_idx[img_id], cfg["input_size"])
    h, w = inp.shape[:2]
    disp = inp
    if img_id in display_idx and display_idx[img_id] != input_idx[img_id]:
      disp = np.asarray(Image.fromarray(load_rgb(display_idx[img_id], None))
                        .resize((w, h), Image.BILINEAR))
    tensor = to_tensor(inp, device)
    probs, cams = {}, {}
    for k in MODEL_KEYS:
      probs[k], cams[k] = runners[k](tensor)

    # descrittori sempre a FEATURE_SIZE x FEATURE_SIZE (soglie confrontabili)
    fs = FEATURE_SIZE
    inp_f = inp if (h, w) == (fs, fs) else np.asarray(
        Image.fromarray(inp).resize((fs, fs), Image.BILINEAR))
    cams_f = {k: c if c.shape == (fs, fs) else
              cv2.resize(c, (fs, fs), interpolation=cv2.INTER_AREA)
              for k, c in cams.items()}
    mask_f, seg_info = segment_lesion(inp_f)
    mask = mask_f if (h, w) == (fs, fs) else cv2.resize(
        mask_f.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    features = compute_case_features(inp_f, mask_f, seg_info, cams_f, probs,
                                     args.threshold, cfg["name"])
    warns = rule_warnings(features)
    features["avvertenze_calcolate"] = [w["testo"] for w in warns]

    try:
      expl = explainer.explain(features)
    except Exception as e:  # noqa: BLE001
      print(f"[{img_id}] LLM non disponibile ({e}); uso il template.")
      expl = TemplateExplainer().explain(features)
    sec = expl["sezioni"]
    ver = verify_explanation(sec, features, warns)

    # avvertenze a regole omesse dall'LLM: mostrate comunque
    missing = set(ver["avvertenze_mancanti"])
    extra = [w["testo"] for w in warns
             if (w["codice"] + (f"@{w['modello']}" if w["modello"] else "")) in missing]
    if split == "training":
      extra.insert(0, f"Immagine del fold {img_fold}, usata nel TRAINING dei "
                      f"checkpoint del fold {ck_fold}: spiegazione non rappresentativa.")

    info = {
        "image_id": img_id, "ground_truth": gt, "soglia": args.threshold,
        "prob": probs,
        "esito": {k: features["modelli"][k]["predizione"]["esito"] for k in MODEL_KEYS},
    }
    src = ("GPT-OSS-20B (Groq)" if expl["meta"]["explainer"] == "llm"
           else "template a regole")
    overlay_img = disp if args.overlay_on == "display" else inp
    input_desc = cfg["input_desc"]
    overlay_desc = ("all'immagine del primo pannello" if args.overlay_on == "display"
                    else "all'input del modello")
    footer = (
        f"Spiegazione generata automaticamente ({src}) a partire da descrittori "
        "quantitativi delle mappe Grad-CAM++; non costituisce una diagnosi. "
        f"Input dei modelli: {input_desc}. Mappe sovrapposte "
        f"{overlay_desc}. "
        "Contorno tratteggiato: segmentazione automatica usata per i descrittori. "
        f"{cfg['name']}, checkpoint del fold {ck_fold}, immagine di {split}. "
        "La ground truth non e' fornita al generatore di testo."
    )
    stem = f"{args.dataset}_nl_gradcam_{n:02d}_{img_id}_{gt}"
    fig_path = os.path.join(fig_dir, stem + ".png")
    render_case_figure(fig_path, disp, overlay_img, cams, mask, info, sec,
                       extra_warnings=extra, footer=footer, dpi=args.dpi,
                       show_contour=not args.no_contour)

    record = {
        "image_id": img_id, "ground_truth": gt, "dataset": cfg["name"],
        "fold_immagine": img_fold, "fold_checkpoint": ck_fold, "split": split,
        "checkpoint": runners["ckpt"],
        "input": {"file": input_idx[img_id], "dimensione_hw": [h, w],
                  "descrizione": input_desc},
        "target_layer": {k: runners[k].target_layer_name for k in MODEL_KEYS},
        "segmentazione": seg_info,
        "descrittori_inviati": features,
        "avvertenze_regole": warns,
        "spiegazione": expl,
        "avvertenze_aggiunte": extra,
        "verifica": ver,
        "figura": fig_path,
    }
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
      json.dump(record, f, ensure_ascii=False, indent=1, default=float)

    m = features["modelli"]
    summary.append({
        "image_id": img_id, "ground_truth": gt, "fold_immagine": img_fold,
        "fold_checkpoint": ck_fold, "split": split,
        "p_resnet50": round(probs["resnet50"], 4),
        "esito_resnet50": m["resnet50"]["predizione"]["esito"],
        "p_mobilevit_s": round(probs["mobilevit_s"], 4),
        "esito_mobilevit_s": m["mobilevit_s"]["predizione"]["esito"],
        "energia_lesione_resnet50_pct": m["resnet50"]["mappa"].get("energia_dentro_lesione_pct"),
        "energia_lesione_mobilevit_s_pct": m["mobilevit_s"]["mappa"].get("energia_dentro_lesione_pct"),
        "iou_top20": features["confronto_modelli"]["iou_regioni_top20"],
        "segmentazione_affidabile": seg_info.get("affidabile"),
        "avvertenze_regole": ";".join(w["codice"] for w in warns),
        "explainer": expl["meta"]["explainer"],
        "verifica_superata": ver["superata"],
        "numeri_non_verificati": " ".join(f"{x:g}" for x in ver["numeri_non_verificati"]),
        "avvertenze_mancanti": ";".join(ver["avvertenze_mancanti"]),
        "termini_vietati": ";".join(ver["termini_vietati"]),
        "latenza_s": expl["meta"].get("latenza_s"),
        "token_totali": (expl["meta"].get("token") or {}).get("totale"),
        "figura": fig_path,
    })
    print(f"[{n}/{len(samples)}] {img_id} ({gt}, fold {img_fold}, {split}) -> "
          f"ResNet {probs['resnet50']:.3f}, MobileViT {probs['mobilevit_s']:.3f} | "
          f"verifica {'OK' if ver['superata'] else 'NON superata'} | {fig_path}")

  if summary:
    path = os.path.join(out_dir, f"{args.dataset}_nl_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
      wr = csv.DictWriter(f, fieldnames=list(summary[0]))
      wr.writeheader()
      wr.writerows(summary)
    print(f"\nRiepilogo: {path}")


if __name__ == "__main__":
  main()
