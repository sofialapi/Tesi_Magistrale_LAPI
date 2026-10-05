"""Test offline (senza GPU, torch o API): descrittori, verifica e figura.

  python -m XAI_NL.nl_selftest --out outputs/xai_nl/selftest

Crea un'immagine sintetica con lesione e vignettatura, due mappe fittizie
(una sulla lesione, una sulla cute) e produce la figura con il template e con
un testo di esempio in stile LLM, per controllare impaginazione e avvertenze.
"""
import argparse
import json
import os

import cv2
import numpy as np

from .nl_cam_features import compute_case_features, rule_warnings, segment_lesion
from .nl_explainer import TemplateExplainer
from .nl_figure import render_case_figure
from .nl_verify import verify_explanation


def synthetic_case(size=224):
  rng = np.random.default_rng(0)
  img = np.full((size, size, 3), (232, 196, 190), np.float32)  # cute rosata
  yy, xx = np.mgrid[:size, :size]
  # lesione irregolare marrone con zona scura e area bluastra
  r = 62 + 10 * np.sin(np.arctan2(yy - 112, xx - 108) * 5)
  les = np.hypot(xx - 108, (yy - 112) * 1.15) < r
  img[les] = (150, 95, 70)
  img[np.hypot(xx - 95, yy - 95) < 22] = (70, 40, 35)
  img[np.hypot(xx - 130, yy - 135) < 14] = (95, 100, 135)
  img += rng.normal(0, 6, img.shape)
  img[np.hypot(xx - 112, yy - 112) > 150] = 5  # vignettatura agli angoli
  img = np.clip(cv2.GaussianBlur(img, (5, 5), 0), 0, 255).astype(np.uint8)

  def blob(cx, cy, s):
    c = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    return (c / c.max()).astype(np.float32)

  cams = {"resnet50": blob(97, 97, 22),                      # sulla zona scura
          "mobilevit_s": 0.35 * blob(110, 115, 45) + blob(200, 205, 16)}  # cute
  cams["mobilevit_s"] /= cams["mobilevit_s"].max()
  probs = {"resnet50": 0.962, "mobilevit_s": 0.541}
  return img, cams, probs


SAMPLE_LLM = {
    "sintesi": "Entrambi i modelli classificano la lesione come melanoma. ResNet-50 "
               "lo fa con sicurezza elevata, MobileViT-S con una probabilita' "
               "vicina alla soglia.",
    "resnet50": "La mappa suggerisce che il modello abbia dato peso soprattutto alla "
                "porzione superiore sinistra della lesione, in una regione focale piu' "
                "scura del resto della lesione. La quasi totalita' dell'attivazione "
                "ricade sulla lesione.",
    "mobilevit_s": "La massima attivazione cade sulla cute distante dalla lesione, in "
                   "basso a destra vicino al bordo dell'immagine. Sulla lesione e' "
                   "presente solo un'attivazione diffusa di bassa intensita'.",
    "confronto": "Le regioni piu' attive dei due modelli sono scarsamente sovrapposte: "
                 "i modelli concordano sull'esito ma non sull'evidenza utilizzata.",
    "avvertenze": [
        "MobileViT-S: buona parte dell'attivazione cade fuori dalla lesione; la "
        "predizione potrebbe basarsi su elementi non clinicamente rilevanti.",
        "MobileViT-S: probabilita' vicina alla soglia, esito poco stabile.",
    ],
}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="outputs/xai_nl/selftest")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  img, cams, probs = synthetic_case()
  mask, seg = segment_lesion(img)
  features = compute_case_features(img, mask, seg, cams, probs, 0.5, "SINTETICO")
  warns = rule_warnings(features)
  features["avvertenze_calcolate"] = [w["testo"] for w in warns]

  info = {"image_id": "SINTETICO_001", "ground_truth": "Melanoma", "soglia": 0.5,
          "prob": probs,
          "esito": {k: features["modelli"][k]["predizione"]["esito"] for k in probs}}
  footer = ("Spiegazione generata automaticamente a partire da descrittori "
            "quantitativi delle mappe Grad-CAM++; non costituisce una diagnosi.")

  results = {}
  for name, sections in (("template", TemplateExplainer().explain(features)["sezioni"]),
                         ("llm_esempio", SAMPLE_LLM)):
    ver = verify_explanation(sections, features, warns)
    missing = set(ver["avvertenze_mancanti"])
    extra = [w["testo"] for w in warns
             if (w["codice"] + (f"@{w['modello']}" if w["modello"] else "")) in missing]
    path = render_case_figure(os.path.join(args.out, f"selftest_{name}.png"),
                              img, img, cams, mask, info, sections,
                              extra_warnings=extra, footer=footer, dpi=150)
    results[name] = {"verifica": ver, "figura": path}

  with open(os.path.join(args.out, "selftest_features.json"), "w",
            encoding="utf-8") as f:
    json.dump({"segmentazione": seg, "descrittori": features,
               "avvertenze": warns, "risultati": results}, f,
              ensure_ascii=False, indent=1)
  print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
  main()
