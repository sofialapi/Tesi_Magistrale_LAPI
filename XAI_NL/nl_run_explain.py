"""Spiegazioni in linguaggio naturale delle mappe Grad-CAM++ (BCN20000 e HAM10000).

Da lanciare dalla root del progetto (Tesi_Magistrale_LAPI):

  # casi delle figure 6.1 e 6.2 della tesi
  python -m XAI_NL.nl_run_explain --dataset bcn --ids ISIC_0000002
  python -m XAI_NL.nl_run_explain --dataset ham --ids ISIC_0028086

  # 3 melanomi + 3 benigni casuali dal fold di validazione
  python -m XAI_NL.nl_run_explain --dataset ham --fold 1 --num_samples 6

  # senza API (testo da template): utile per controllare figure e descrittori
  python -m XAI_NL.nl_run_explain --dataset ham --ids ISIC_0028086 --no_llm
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

from .nl_cam_features import compute_case_features, rule_warnings, segment_lesion
from .nl_config import (DATASETS, FIGURE_DIR, LLM_MODEL, MODEL_KEYS,
                        OUTPUT_DIR)
from .nl_explainer import GroqExplainer, TemplateExplainer
from .nl_figure import render_case_figure
from .nl_verify import verify_explanation

ID_COLS = ("image_id", "image_name", "isic_id", "image")
TARGET_COLS = ("target", "MEL", "label")


def parse_args():
  ap = argparse.ArgumentParser(description="Grad-CAM++ + spiegazione NL (GPT-OSS-20B)")
  ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
  ap.add_argument("--csv")
  ap.add_argument("--processed_dir", help="immagini preelaborate (input dei modelli)")
  ap.add_argument("--raw_dir", help="immagini originali (pannello per il clinico)")
  ap.add_argument("--cnn_ckpt")
  ap.add_argument("--hybrid_ckpt")
  ap.add_argument("--threshold", type=float, default=0.5)
  ap.add_argument("--ids", nargs="+", help="ID specifici (es. ISIC_0028086)")
  ap.add_argument("--fold", type=int, help="fold di validazione del checkpoint")
  ap.add_argument("--fold_col", default="fold")
  ap.add_argument("--num_samples", type=int, default=6)
  ap.add_argument("--selection", choices=["random", "first"], default="random")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--overlay_on", choices=["processed", "raw"], default="processed",
                  help="immagine su cui sovrapporre le mappe")
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


def normalize_id(x):
  s = os.path.basename(str(x))
  if s.lower().endswith((".jpg", ".jpeg", ".png")):
    s = os.path.splitext(s)[0]
  return s.replace("_downsampled", "")


def build_index(root):
  idx = {}
  if not root or not os.path.isdir(root):
    return idx
  for dp, _, files in os.walk(root):
    for f in files:
      if f.lower().endswith((".jpg", ".jpeg", ".png")):
        idx.setdefault(normalize_id(f), os.path.join(dp, f))
  return idx


def detect_columns(df):
  id_col = next((c for c in ID_COLS if c in df.columns), None)
  if id_col is None:
    raise KeyError(f"Nessuna colonna ID tra {ID_COLS}: {list(df.columns)}")
  tgt = next((c for c in TARGET_COLS if c in df.columns), None)
  if tgt is None and "dx" in df.columns:
    df["target"] = (df["dx"] == "mel").astype(int)
    tgt = "target"
  if tgt is None:
    raise KeyError(f"Nessuna colonna target tra {TARGET_COLS} o 'dx'")
  return id_col, tgt


def select_samples(df, args, id_col, tgt, proc_idx):
  df = df.copy()
  df["_id"] = df[id_col].map(normalize_id)
  df = df[df["_id"].isin(proc_idx)]  # solo casi con immagine preelaborata
  if args.ids:
    wanted = [normalize_id(i) for i in args.ids]
    sel = df[df["_id"].isin(wanted)]
    missing = sorted(set(wanted) - set(sel["_id"]))
    if missing:
      print(f"[AVVISO] ID non trovati (CSV o immagini preelaborate): {missing}")
    return sel.drop_duplicates("_id")

  if args.fold is not None:
    if args.fold_col in df.columns:
      df = df[df[args.fold_col] == args.fold]
    else:
      print(f"[AVVISO] colonna '{args.fold_col}' assente: impossibile filtrare il fold.")
  else:
    print("[AVVISO] nessun fold indicato: i casi potrebbero appartenere al "
          "training set del checkpoint. Per la tesi usa --fold o --ids di validazione.")

  n_mal = args.num_samples // 2
  mal, ben = df[df[tgt] == 1], df[df[tgt] == 0]
  if args.selection == "random":
    mal = mal.sample(min(n_mal, len(mal)), random_state=args.seed)
    ben = ben.sample(min(args.num_samples - len(mal), len(ben)), random_state=args.seed)
  else:
    mal = mal.head(n_mal)
    ben = ben.head(args.num_samples - len(mal))
  return pd.concat([mal, ben])


def main():
  args = parse_args()
  cfg = dict(DATASETS[args.dataset])
  for k in ("csv", "processed_dir", "raw_dir", "cnn_ckpt", "hybrid_ckpt"):
    if getattr(args, k):
      cfg[k] = getattr(args, k)
  if args.fold is None and not args.ids:
    args.fold = cfg["default_fold"]

  fig_dir = args.figure_dir or os.path.join(FIGURE_DIR, args.dataset)
  out_dir = args.output_dir or os.path.join(OUTPUT_DIR, args.dataset)
  os.makedirs(fig_dir, exist_ok=True)
  os.makedirs(out_dir, exist_ok=True)

  # import qui: torch serve solo per l'esecuzione completa
  import torch
  from .nl_models import CamRunner, load_rgb, to_tensor

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  runners = {
      "resnet50": CamRunner("resnet50", cfg["cnn_ckpt"], device),
      "mobilevit_s": CamRunner("mobilevit_s", cfg["hybrid_ckpt"], device),
  }
  for k, r in runners.items():
    print(f"[{k}] target layer Grad-CAM++: {r.target_layer_name}")

  if args.no_llm:
    explainer = TemplateExplainer()
  else:
    explainer = GroqExplainer(
        model=args.llm_model, reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        cache_dir=os.path.join(OUTPUT_DIR, "cache"))

  proc_idx = build_index(cfg["processed_dir"])
  raw_idx = build_index(cfg["raw_dir"])
  print(f"Immagini indicizzate: {len(proc_idx)} preelaborate, {len(raw_idx)} raw")
  if not proc_idx:
    raise SystemExit(f"Nessuna immagine in {cfg['processed_dir']}")

  df = pd.read_csv(cfg["csv"])
  id_col, tgt = detect_columns(df)
  samples = select_samples(df, args, id_col, tgt, proc_idx)
  print(f"Casi selezionati: {len(samples)} | soglia tau={args.threshold} | "
        f"explainer={explainer.name}")

  summary = []
  for n, (_, row) in enumerate(samples.iterrows(), start=1):
    img_id = row["_id"]
    gt = "Melanoma" if int(row[tgt]) == 1 else "Benigno"
    proc = load_rgb(proc_idx[img_id])
    raw = load_rgb(raw_idx[img_id]) if img_id in raw_idx else None
    tensor = to_tensor(proc, device)

    probs, cams = {}, {}
    for k in MODEL_KEYS:
      probs[k], cams[k] = runners[k](tensor)

    mask, seg_info = segment_lesion(proc)
    features = compute_case_features(proc, mask, seg_info, cams, probs,
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

    # le avvertenze a regole omesse dall'LLM vengono comunque mostrate
    missing = set(ver["avvertenze_mancanti"])
    extra = [w["testo"] for w in warns
             if (w["codice"] + (f"@{w['modello']}" if w["modello"] else "")) in missing]

    info = {
        "image_id": img_id, "ground_truth": gt, "soglia": args.threshold,
        "prob": probs,
        "esito": {k: features["modelli"][k]["predizione"]["esito"] for k in MODEL_KEYS},
    }
    src = ("GPT-OSS-20B (Groq)" if expl["meta"]["explainer"] == "llm"
           else "template a regole")
    overlay_img = raw if (args.overlay_on == "raw" and raw is not None) else proc
    footer = (
        f"Spiegazione generata automaticamente ({src}) a partire da descrittori "
        "quantitativi delle mappe Grad-CAM++; non costituisce una diagnosi. "
        f"Mappe sovrapposte all'immagine {'originale' if overlay_img is raw else 'preelaborata'}"
        " fornita ai modelli. Contorno tratteggiato: segmentazione automatica "
        "della lesione usata per i descrittori. Dataset: "
        f"{cfg['name']}. La ground truth non e' fornita al generatore di testo."
    )
    stem = f"{args.dataset}_nl_gradcam_{n:02d}_{img_id}_{gt}"
    fig_path = os.path.join(fig_dir, stem + ".png")
    render_case_figure(
        fig_path, raw if raw is not None else proc, overlay_img, cams, mask,
        info, sec, extra_warnings=extra, footer=footer, dpi=args.dpi,
        show_contour=not args.no_contour)

    record = {
        "image_id": img_id, "ground_truth": gt, "dataset": cfg["name"],
        "checkpoint": {"resnet50": cfg["cnn_ckpt"], "mobilevit_s": cfg["hybrid_ckpt"]},
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
        "image_id": img_id, "ground_truth": gt,
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
    print(f"[{n}/{len(samples)}] {img_id} ({gt}) -> ResNet {probs['resnet50']:.3f}, "
          f"MobileViT {probs['mobilevit_s']:.3f} | verifica "
          f"{'OK' if ver['superata'] else 'NON superata'} | {fig_path}")

  if summary:
    path = os.path.join(out_dir, f"{args.dataset}_nl_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
      wr = csv.DictWriter(f, fieldnames=list(summary[0]))
      wr.writeheader()
      wr.writerows(summary)
    print(f"\nRiepilogo: {path}")


if __name__ == "__main__":
  main()
