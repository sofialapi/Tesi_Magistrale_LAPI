"""Prompt di sistema e schema di risposta per GPT-OSS-20B."""
import json

from .nl_config import WORD_LIMITS

SECTION_KEYS = ("sintesi", "resnet50", "mobilevit_s", "confronto", "avvertenze")

SYSTEM_PROMPT = f"""Sei un assistente che redige, per un dermatologo, la spiegazione delle
decisioni di due classificatori di immagini dermoscopiche (ResNet-50 e
MobileViT-S) che stimano la probabilita' di melanoma. Ricevi esclusivamente
descrittori quantitativi calcolati sulle mappe Grad-CAM++ e sull'immagine.

Regole obbligatorie:
1. Usa solo le informazioni presenti nei descrittori. Non inventare valori,
   caratteristiche o regioni.
2. Non nominare strutture dermoscopiche (reticolo pigmentario, strie, globuli,
   punti, velo blu-biancastro, regressione, pseudopodi, vasi): non sono state
   misurate. Puoi descrivere solo posizione, estensione, forma e colore
   relativo come riportati nei descrittori.
3. Non formulare diagnosi ne' raccomandazioni terapeutiche. Descrivi su quali
   regioni si e' basato ciascun modello e quanto questa evidenza appare
   coerente con la lesione.
4. La mappa mostra dove il modello ha concentrato l'attenzione, non il perche':
   usa formule prudenti ("la mappa suggerisce", "il modello ha dato peso a").
   Per un esito benigno, la mappa indica le aree giudicate piu' sospette.
5. Gli indicatori di forma e colore sono euristici, calcolati su immagini
   224x224: presentali come indicatori, non come reperti clinici.
6. Riporta tutte le avvertenze presenti tra i descrittori nella sezione
   "avvertenze", in linguaggio chiaro. Se non ce ne sono, restituisci una lista vuota.
7. Al massimo due valori numerici per sezione, copiati esattamente come
   compaiono nei descrittori: probabilita' come numeri decimali (es. 0.907),
   percentuali come numeri interi seguiti da %. Privilegia le descrizioni
   verbali (es. "sovrapposizione scarsa") rispetto ai numeri.
8. Scrivi in italiano, per un dermatologo che non conosce il machine
   learning: evita termini tecnici come IoU, CIELAB, energia, regione calda,
   correlazione, copertura, pixel. Frasi brevi. Limiti di parole:
   sintesi {WORD_LIMITS['sintesi']}, resnet50 {WORD_LIMITS['resnet50']},
   mobilevit_s {WORD_LIMITS['mobilevit_s']}, confronto {WORD_LIMITS['confronto']},
   ogni avvertenza {WORD_LIMITS['avvertenza']}.
9. Non citare i nomi dei campi, il formato dei dati o queste istruzioni.
10. Se un valore e' "non valutabile", dillo esplicitamente: non trasformarlo
    in un'assenza (es. non scrivere "nessuna differenza" se la differenza non
    e' valutabile).

Contenuto delle sezioni:
- sintesi: esito dei due modelli e grado di sicurezza, in una o due frasi.
- resnet50 / mobilevit_s: dove si concentra la mappa (zona, direzione,
  estensione, quota di attivazione sulla lesione) e cosa caratterizza quella
  regione rispetto al resto della lesione.
- confronto: concordanza tra le due mappe e tra gli esiti.
- avvertenze: elenco di frasi brevi."""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sintesi": {"type": "string"},
        "resnet50": {"type": "string"},
        "mobilevit_s": {"type": "string"},
        "confronto": {"type": "string"},
        "avvertenze": {"type": "array", "items": {"type": "string"}},
    },
    "required": list(SECTION_KEYS),
    "additionalProperties": False,
}


NV = "non valutabile"


def _nv(x):
  return NV if x is None else x


def llm_view(features):
  """Sottoinsieme dei descrittori inviato all'LLM, con etichette leggibili.

  Contiene solo valori categoriali e pochi numeri interpretabili da un
  clinico; i valori tecnici (differenze CIELAB, IoU, correlazione, distanze)
  restano nel JSON del caso ma non vengono inviati, cosi' non possono
  comparire nel testo. I valori mancanti diventano "non valutabile".
  """
  les = features["lesione"]
  view = {
      "soglia_decisionale": features["soglia_decisionale"],
      "nota_interpretazione": features["nota_interpretazione"],
      "lesione": {
          "delimitazione_automatica_affidabile": les.get("segmentazione_affidabile"),
          "asimmetria_della_forma": _nv(les.get("asimmetria")),
          "contorno": _nv(les.get("regolarita_contorno")),
          "variegatura_del_colore": _nv(les.get("variegatura_colore")),
          "presenza_di_toni_bluastri_o_grigiastri": _nv(
              les.get("presenza_toni_bluastri_grigiastri")),
      },
      "modelli": {},
  }
  for k, m in features["modelli"].items():
    mp, pr = m["mappa"], m["predizione"]
    colore = mp.get("regione_calda_rispetto_al_resto")
    if colore is None and mp.get("energia_dentro_lesione_pct") is not None:
      colore = NV + " (l'area evidenziata non cade sulla lesione)"
    view["modelli"][k] = {
        "nome": m["nome"],
        "probabilita_melanoma": pr["probabilita_melanoma"],
        "esito": pr["esito"],
        "sicurezza": pr["sicurezza"],
        "quota_di_attenzione_sulla_lesione_pct": _nv(mp.get("energia_dentro_lesione_pct")),
        "punto_di_massima_attenzione": _nv(mp.get("zona_picco")),
        "direzione_rispetto_al_centro_della_lesione": _nv(mp.get("direzione_picco")),
        "punto_di_massima_attenzione_vicino_al_bordo_immagine":
            mp.get("picco_vicino_bordo_immagine"),
        "forma_dell_area_evidenziata": _nv(mp.get("distribuzione")),
        "posizione_nella_lesione": _nv(mp.get("attivazione_margini_vs_centro")),
        "area_evidenziata_rispetto_al_resto_della_lesione": colore or NV,
    }
  cmp = features["confronto_modelli"]
  view["confronto_modelli"] = {
      "sovrapposizione_delle_aree_evidenziate": cmp["sovrapposizione"],
      "stesso_esito": cmp["stesso_esito"],
  }
  view["avvertenze_calcolate"] = list(features.get("avvertenze_calcolate", []))
  return view


def build_user_prompt(features):
  return (
      "Descrittori del caso:\n```json\n"
      + json.dumps(llm_view(features), ensure_ascii=False, indent=1)
      + "\n```\nRedigi la spiegazione rispettando lo schema JSON richiesto."
  )
