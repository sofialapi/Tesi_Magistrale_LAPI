"""Descrittori, avvertenze, prompt, generazione e verifica del testo della scheda SHAP.

Stessa impostazione della Sezione 6.2 (XAI_NL): GPT-OSS-20B via Groq riceve solo
descrittori gia' calcolati (probabilita', contributi SHAP dei dati clinici,
indicatore della mappa Grad-CAM++), mai l'immagine ne' la verita'; le avvertenze
sono calcolate a regole e il testo viene verificato automaticamente.
"""
import hashlib
import json
import math
import os
import re
import time

import numpy as np

from XAI_NL.nl_verify import _IGNORE, _NUM, _collect_numbers, _matches

LLM_MODEL = "openai/gpt-oss-20b"
PROMPT_VERSION = "shap-v2"  # cambiala quando modifichi il prompt: invalida la cache

THRESHOLD = 0.5
# Intensita' di un contributo SHAP (|phi| in log-odds). Riferimento: la media di
# |phi| per variabile sull'intera coorte va da circa 0.05 a 0.2 (Sez. 6.3).
INTENSITY = ((0.5, "forte"), (0.2, "moderato"), (0.05, "lieve"))
CLINIC_NEGLIGIBLE = 0.2       # |somma dei contributi clinici| sotto: effetto trascurabile
SEX_WARN = 0.10               # |phi_sesso| sopra: avvertenza "relazione non confermata"
SMALL_AREA_MM2 = 2.0          # area sotto: anomalia nota del dataset
NEAR_THRESHOLD_MARGIN = 0.10  # |p - soglia| sotto: stima vicina alla soglia
MIN_LESION_ENERGY = 0.50      # quota di attenzione sulla lesione sotto: avvertenza

WORD_LIMITS = {"sintesi": 45, "immagine": 45, "dati_clinici": 70, "avvertenza": 30}
SECTION_KEYS = ("sintesi", "immagine", "dati_clinici", "avvertenze")

SHORT_NAMES = {"tbp_lv_areaMM2": "Area", "tbp_lv_eccentricity": "Allungamento",
               "tbp_lv_symm_2axis": "Asimmetria", "age_approx": "Età", "sex": "Sesso",
               "anatom_site_general": "Sede"}
VAR_NAMES = {
    "tbp_lv_areaMM2": "Area della lesione",
    "tbp_lv_eccentricity": "Allungamento della forma (eccentricità)",
    "tbp_lv_symm_2axis": "Asimmetria della forma",
    "age_approx": "Età",
    "sex": "Sesso",
    "anatom_site_general": "Sede anatomica",
}
SITE_IT = {"posterior torso": "tronco posteriore", "anterior torso": "tronco anteriore",
           "lower extremity": "arto inferiore", "upper extremity": "arto superiore",
           "head/neck": "testa/collo"}
SEX_IT = {"male": "maschio", "female": "femmina"}


def sigmoid(z):
  return 1.0 / (1.0 + math.exp(-z))


def _missing(v):
  return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).lower() == "nan"


def format_value(var, v):
  """Valore della variabile in unita' cliniche (stringa)."""
  if _missing(v):
    return "non disponibile"
  if var == "tbp_lv_areaMM2":
    d = 2 * math.sqrt(float(v) / math.pi)
    return f"{float(v):.1f} mm² (diametro equivalente {d:.1f} mm)"
  if var == "age_approx":
    return f"{float(v):.0f} anni"
  if var == "tbp_lv_eccentricity":
    v = float(v)
    q = "quasi circolare" if v < 0.6 else ("moderatamente allungata" if v < 0.85 else "allungata")
    return f"{v:.2f} (forma {q})"
  if var == "tbp_lv_symm_2axis":
    v = float(v)
    q = "bassa" if v < 0.2 else ("media" if v < 0.35 else "alta")
    return f"{v:.2f} ({q})"
  if var == "sex":
    return SEX_IT.get(v, str(v))
  if var == "anatom_site_general":
    return SITE_IT.get(v, str(v))
  return str(v)


def short_value(var, v):
  """Valore compatto per le etichette del grafico."""
  if _missing(v):
    return "n.d."
  if var == "tbp_lv_areaMM2":
    return f"{float(v):.1f} mm² (Ø {2 * math.sqrt(float(v) / math.pi):.1f} mm)"
  if var in ("tbp_lv_eccentricity", "tbp_lv_symm_2axis"):
    return f"{float(v):.2f}"
  return format_value(var, v)


def intensity(phi):
  a = abs(phi)
  for thr, name in INTENSITY:
    if a >= thr:
      return name
  return "trascurabile"


def outcome(p, tau=THRESHOLD):
  return "maligna" if p >= tau else "benigna"


def confidence(p, tau=THRESHOLD):
  d = abs(p - tau)
  return "alta" if d >= 0.4 else ("moderata" if d >= 0.2 else "bassa")


# ----------------------------------------------------------------------------
# Descrittori del caso
# ----------------------------------------------------------------------------
def case_features(row, var_order, model_label, cam_info, tau=THRESHOLD):
  """Descrittori completi del caso (salvati nel JSON).

  row: riga di shap_values_<case>.csv (phi_*, val_*, logit, base_tab, v_none).
  cam_info: {"quota_lesione_pct": int|None, "segmentazione_affidabile": bool|None}
  """
  p_ref, p_img, p_fin = sigmoid(row["v_none"]), sigmoid(row["base_tab"]), sigmoid(row["logit"])
  delta = row["logit"] - row["base_tab"]
  if outcome(p_img, tau) != outcome(p_fin, tau):
    ruolo = "ribaltano l'esito"
  elif abs(delta) < CLINIC_NEGLIGIBLE:
    ruolo = "effetto trascurabile"
  elif (delta > 0) == (row["logit"] >= 0):
    ruolo = "rafforzano l'esito"
  else:
    ruolo = "attenuano l'esito"

  fattori = []
  for var in var_order:
    phi = float(row[f"phi_{var}"])
    fattori.append({
        "codice": var,
        "variabile": VAR_NAMES[var],
        "nome_breve": SHORT_NAMES[var],
        "valore": format_value(var, row[f"val_{var}"]),
        "valore_breve": short_value(var, row[f"val_{var}"]),
        "dato_mancante": _missing(row[f"val_{var}"]),
        "effetto": "verso malignità" if phi > 0 else "verso benignità",
        "intensita": intensity(phi),
        "phi": round(phi, 3),
    })
  fattori.sort(key=lambda f: -abs(f["phi"]))

  return {
      "modello": model_label,
      "soglia_decisionale": tau,
      "probabilita": {
          "riferimento": round(p_ref, 3),
          "solo_immagine": round(p_img, 3),
          "immagine_e_dati_clinici": round(p_fin, 3),
      },
      "esito": outcome(p_fin, tau),
      "esito_solo_immagine": outcome(p_img, tau),
      "sicurezza": confidence(p_fin, tau),
      "ruolo_dati_clinici": ruolo,
      "fattori_clinici": fattori,
      "mappa_immagine": cam_info,
      "val_area": None if _missing(row["val_tbp_lv_areaMM2"]) else float(row["val_tbp_lv_areaMM2"]),
  }


def rule_warnings(f, tau=THRESHOLD):
  """Avvertenze deterministiche: lista di {codice, testo}."""
  w = []
  if f["ruolo_dati_clinici"] == "ribaltano l'esito":
    w.append({"codice": "DECISIONE_DA_CLINICA", "testo":
              f"Con la sola immagine la lesione sarebbe stata classificata come "
              f"{f['esito_solo_immagine']}: l'esito dipende dai dati clinici, di cui va "
              "verificata la correttezza."})
  sex = next(x for x in f["fattori_clinici"] if x["codice"] == "sex")
  if abs(sex["phi"]) >= SEX_WARN and not sex["dato_mancante"]:
    w.append({"codice": "SESSO_NON_CONFERMATO", "testo":
              "Il modello ha dato peso al sesso, ma la relazione appresa è opposta alla "
              "frequenza di lesioni maligne osservata nei dati: contributo da considerare con cautela."})
  miss = [x["variabile"].lower() for x in f["fattori_clinici"] if x["dato_mancante"]]
  if miss:
    w.append({"codice": "DATO_MANCANTE", "testo":
              f"Dato non disponibile ({', '.join(miss)}), sostituito con il valore più frequente: "
              "il relativo contributo non descrive il paziente."})
  if f["val_area"] is not None and f["val_area"] <= SMALL_AREA_MM2:
    w.append({"codice": "AREA_PICCOLA", "testo":
              "Area inferiore a 2 mm²: in questa fascia i dati di addestramento presentano "
              "un'anomalia nota, il contributo dell'area è poco affidabile."})
  if abs(f["probabilita"]["immagine_e_dati_clinici"] - tau) < NEAR_THRESHOLD_MARGIN:
    w.append({"codice": "VICINO_SOGLIA", "testo":
              "La probabilità stimata è vicina alla soglia decisionale: esito poco stabile."})
  q = f["mappa_immagine"].get("quota_lesione_pct")
  if q is not None and q < 100 * MIN_LESION_ENERGY:
    w.append({"codice": "FUORI_LESIONE", "testo":
              "La mappa dell'immagine si concentra in buona parte fuori dalla lesione."})
  return w


NV = "non valutabile"


def llm_view(f):
  """Sottoinsieme dei descrittori inviato all'LLM (niente valori SHAP grezzi)."""
  q = f["mappa_immagine"].get("quota_lesione_pct")
  return {
      "modello": f["modello"],
      "soglia_decisionale": f["soglia_decisionale"],
      "probabilita_di_malignita": {
          "lesione_di_riferimento": f["probabilita"]["riferimento"],
          "con_la_sola_immagine": f["probabilita"]["solo_immagine"],
          "con_immagine_e_dati_clinici": f["probabilita"]["immagine_e_dati_clinici"],
      },
      "esito": f["esito"],
      "sicurezza": f["sicurezza"],
      "effetto_dei_dati_clinici_sull_esito": f["ruolo_dati_clinici"],
      "dati_clinici_in_ordine_di_peso": [
          {"dato": x["variabile"], "valore": x["valore"], "spinge": x["effetto"],
           "intensita": x["intensita"]}
          for x in f["fattori_clinici"]],
      "mappa_di_attenzione_sull_immagine": {
          "quota_di_attenzione_sulla_lesione_pct": NV if q is None else q,
          "nota": "mappa a bassa risoluzione, indica solo la zona approssimativa",
      },
      "avvertenze_calcolate": list(f.get("avvertenze_calcolate", [])),
  }


SYSTEM_PROMPT = f"""Sei un assistente che redige, per un dermatologo, la spiegazione della
decisione di un classificatore che stima la probabilita' che una lesione cutanea,
fotografata con Total Body Photography, sia maligna. Il classificatore combina
l'immagine della lesione con alcuni dati clinici. Ricevi esclusivamente
descrittori gia' calcolati: le probabilita' stimate (per una lesione di
riferimento, con la sola immagine e con immagine e dati clinici), il peso e la
direzione di ciascun dato clinico e un indicatore sulla mappa di attenzione.

Regole obbligatorie:
1. Usa solo le informazioni presenti nei descrittori. Non inventare valori o
   caratteristiche della lesione.
2. Il peso di un dato clinico descrive quanto il modello lo ha usato, non una
   relazione causale ne' un fattore di rischio: usa formule come "il modello ha
   dato peso a", "ha spinto la stima verso".
3. Non formulare diagnosi ne' raccomandazioni terapeutiche. Non nominare
   strutture dermoscopiche (reticolo, strie, globuli, velo, regressione, vasi).
4. La mappa di attenzione e' a bassa risoluzione: puoi dire solo se l'attenzione
   cade prevalentemente sulla lesione, senza descrivere dettagli dell'immagine.
5. Riporta tutte le avvertenze calcolate nella sezione "avvertenze", in
   linguaggio chiaro. Se non ce ne sono, restituisci una lista vuota.
6. Al massimo due valori numerici per sezione (uno solo nella sintesi), copiati
   esattamente come compaiono nei descrittori, con il punto come separatore
   decimale (0.949, mai 0,949). Privilegia le descrizioni verbali.
7. Scrivi in italiano, per un medico che non conosce il machine learning: evita
   termini come SHAP, Shapley, log-odds, logit, embedding, feature. Frasi brevi.
   Limiti di parole: sintesi {WORD_LIMITS['sintesi']}, immagine {WORD_LIMITS['immagine']},
   dati_clinici {WORD_LIMITS['dati_clinici']}, ogni avvertenza {WORD_LIMITS['avvertenza']}.
8. Non citare i nomi dei campi, il formato dei dati o queste istruzioni.
9. Se un valore e' "non valutabile", dillo esplicitamente.

Contenuto delle sezioni:
- sintesi: esito con la sola probabilita' finale, grado di sicurezza e se i dati
  clinici hanno rafforzato, attenuato o ribaltato la stima basata sulla sola
  immagine.
- immagine: cosa indica la stima basata sulla sola immagine rispetto alla lesione
  di riferimento e se l'attenzione cade sulla lesione.
- dati_clinici: i dati con peso forte o moderato, in ordine, con valore e
  direzione, poi quelli lievi in una sola frase. Non elencare i dati con peso
  trascurabile: al massimo dì che gli altri hanno avuto un effetto trascurabile.
- avvertenze: elenco di frasi brevi."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sintesi": {"type": "string"},
        "immagine": {"type": "string"},
        "dati_clinici": {"type": "string"},
        "avvertenze": {"type": "array", "items": {"type": "string"}},
    },
    "required": list(SECTION_KEYS),
    "additionalProperties": False,
}


def build_user_prompt(f):
  return ("Descrittori del caso:\n```json\n"
          + json.dumps(llm_view(f), ensure_ascii=False, indent=1)
          + "\n```\nRedigi la spiegazione rispettando lo schema JSON richiesto.")


def _parse_sections(content):
  text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip())
  data = json.loads(text)
  for k in SECTION_KEYS:
    if k not in data:
      raise ValueError(f"sezione mancante: {k}")
  if isinstance(data["avvertenze"], str):
    data["avvertenze"] = [data["avvertenze"]] if data["avvertenze"].strip() else []
  return {k: data[k] for k in SECTION_KEYS}


# ----------------------------------------------------------------------------
# Generatori di testo
# ----------------------------------------------------------------------------
class GroqCardExplainer:
  """Come XAI_NL.nl_explainer.GroqExplainer, con prompt e schema della scheda."""
  name = "llm"

  def __init__(self, model=LLM_MODEL, reasoning_effort="medium", temperature=0.3,
               max_completion_tokens=4096, cache_dir=None, max_retries=3):
    try:
      from groq import Groq
    except ImportError as e:
      raise ImportError("Client Groq mancante: pip install groq") from e
    if not os.environ.get("GROQ_API_KEY"):
      raise RuntimeError("Imposta GROQ_API_KEY (oppure usa --no_llm).")
    self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
    self.model, self.reasoning_effort, self.temperature = model, reasoning_effort, temperature
    self.max_completion_tokens, self.max_retries, self.cache_dir = (
        max_completion_tokens, max_retries, cache_dir)
    if cache_dir:
      os.makedirs(cache_dir, exist_ok=True)
    self._variants = [
        ("json_schema", {"response_format": {"type": "json_schema", "json_schema": {
            "name": "scheda_shap", "strict": True, "schema": RESPONSE_SCHEMA}},
            "reasoning_effort": reasoning_effort, "include_reasoning": False}),
        ("json_object", {"response_format": {"type": "json_object"},
                         "reasoning_effort": reasoning_effort}),
        ("json_object_base", {"response_format": {"type": "json_object"}}),
    ]
    self._variant_idx = 0

  def _cache_path(self, f):
    blob = json.dumps({"v": PROMPT_VERSION, "m": self.model, "e": self.reasoning_effort,
                       "t": self.temperature, "f": llm_view(f)}, sort_keys=True,
                      ensure_ascii=False)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]
    return os.path.join(self.cache_dir, f"{h}.json") if self.cache_dir else None

  def explain(self, f):
    cache_path = self._cache_path(f)
    if cache_path and os.path.exists(cache_path):
      with open(cache_path, encoding="utf-8") as fh:
        cached = json.load(fh)
      cached["meta"]["cache"] = True
      return cached
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(f)}]
    last_err, attempts = None, 0
    while attempts < self.max_retries and self._variant_idx < len(self._variants):
      vname, extra = self._variants[self._variant_idx]
      t0 = time.time()
      try:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens, **extra)
      except TypeError as e:
        last_err = e
        self._variant_idx += 1
        continue
      except Exception as e:  # noqa: BLE001
        last_err = e
        if getattr(e, "status_code", None) == 400:
          self._variant_idx += 1
          continue
        attempts += 1
        time.sleep(2 ** attempts)
        continue
      try:
        sections = _parse_sections(resp.choices[0].message.content)
      except (ValueError, json.JSONDecodeError) as e:
        last_err = e
        attempts += 1
        continue
      usage = getattr(resp, "usage", None)
      result = {"sezioni": sections, "meta": {
          "explainer": "llm", "modello": self.model, "prompt_version": PROMPT_VERSION,
          "reasoning_effort": self.reasoning_effort, "temperature": self.temperature,
          "response_format": vname, "latenza_s": round(time.time() - t0, 2), "cache": False,
          "token_totali": getattr(usage, "total_tokens", None)}}
      if cache_path:
        with open(cache_path, "w", encoding="utf-8") as fh:
          json.dump(result, fh, ensure_ascii=False, indent=1)
      return result
    raise RuntimeError(f"Generazione LLM fallita: {last_err!r}")


class TemplateCardExplainer:
  """Testo deterministico dagli stessi descrittori (test offline e fallback)."""
  name = "template"

  def explain(self, f):
    p = f["probabilita"]
    ruolo = {"ribaltano l'esito": "hanno ribaltato",
             "rafforzano l'esito": "hanno rafforzato",
             "attenuano l'esito": "hanno attenuato",
             "effetto trascurabile": "hanno modificato in modo trascurabile"}[f["ruolo_dati_clinici"]]
    sintesi = (f"Lesione classificata come {f['esito']} (probabilità {p['immagine_e_dati_clinici']}, sicurezza "
               f"{f['sicurezza']}). I dati clinici {ruolo} la stima basata sulla sola immagine.")
    q = f["mappa_immagine"].get("quota_lesione_pct")
    dove = (NV if q is None else
            ("prevalentemente sulla lesione" if q >= 50 else "in buona parte fuori dalla lesione"))
    immagine = (f"Con la sola immagine la probabilità sarebbe {p['solo_immagine']}, contro "
                f"{p['riferimento']} per una lesione di riferimento. L'attenzione del modello "
                f"cade {dove}." if q is not None else
                f"Con la sola immagine la probabilità sarebbe {p['solo_immagine']}, contro "
                f"{p['riferimento']} per una lesione di riferimento. La posizione "
                "dell'attenzione rispetto alla lesione non è valutabile.")
    rilevanti = [x for x in f["fattori_clinici"] if x["intensita"] in ("forte", "moderato")]
    if rilevanti:
      parti = [f"{x['variabile'].lower()} ({x['valore'].split(' (')[0]}, peso {x['intensita']} "
               f"{x['effetto']})" for x in rilevanti[:3]]
      clin = "Il modello ha dato peso soprattutto a: " + "; ".join(parti) + "."
    else:
      clin = "Nessun dato clinico ha avuto un peso più che lieve sulla stima."
    return {"sezioni": {"sintesi": sintesi, "immagine": immagine, "dati_clinici": clin,
                        "avvertenze": list(f.get("avvertenze_calcolate", []))},
            "meta": {"explainer": "template", "prompt_version": PROMPT_VERSION, "cache": False}}


# ----------------------------------------------------------------------------
# Verifica di fedelta'
# ----------------------------------------------------------------------------
_FORBIDDEN = {
    "reticolo": r"\breticol\w*", "strie": r"\bstri[ea]\b", "globuli": r"\bglobul\w*",
    "velo": r"\bvelo\b", "regressione": r"\bregression\w*", "vasi": r"\bvas[oi]\b|\bvascolar\w*",
    "diagnosi": r"\bdiagnosi\s+di\b|\bsi\s+tratta\s+di\s+(?:un\s+)?(?:melanoma|carcinoma)\b",
}
_JARGON = {"SHAP": r"\bshap\b|shapley", "log-odds": r"log-?odds|\blogit\w*",
           "embedding": r"\bembedding\b", "feature": r"\bfeature\b"}
_WARN_KEYS = {
    "DECISIONE_DA_CLINICA": r"ribalt|oppost|sola immagine|solo immagine|dipende dai dati",
    "SESSO_NON_CONFERMATO": r"\bsesso\b",
    "DATO_MANCANTE": r"mancant|non disponibil|sostituit",
    "AREA_PICCOLA": r"\barea\b|piccol",
    "VICINO_SOGLIA": r"soglia",
    "FUORI_LESIONE": r"fuori|estern|\bcute\b",
}
MAX_NUMBERS_PER_SECTION = 2


def _words(s):
  return len(re.findall(r"\w+", s))


def verify(sections, f, rule_warns):
  allowed = []
  _collect_numbers(llm_view(f), allowed)
  allowed += [224.0, 7.0, 2.0]
  full = " ".join([sections[k] for k in sections if k != "avvertenze"] + list(sections["avvertenze"]))
  numeri = [float(n.replace(",", ".")) for n in _NUM.findall(_IGNORE.sub(" ", full))]
  non_verificati = sorted({n for n in numeri if not _matches(n, allowed)})
  low = full.lower()
  vietati = [k for k, p in _FORBIDDEN.items() if re.search(p, low)]
  gergo = [k for k, p in _JARGON.items() if re.search(p, low)]
  avv = " ".join(sections["avvertenze"]).lower()
  mancanti = [w["codice"] for w in rule_warns
              if not re.search(_WARN_KEYS.get(w["codice"], r"$^"), avv)]
  lunghezze = {k: _words(sections[k]) for k in sections if k != "avvertenze"}
  superati = [k for k, n in lunghezze.items() if n > WORD_LIMITS[k] * 1.2]
  superati += [f"avvertenza_{i+1}" for i, a in enumerate(sections["avvertenze"])
               if _words(a) > WORD_LIMITS["avvertenza"] * 1.2]
  per_sez = {k: len(_NUM.findall(_IGNORE.sub(" ", sections[k])))
             for k in sections if k != "avvertenze"}
  troppi = [k for k, n in per_sez.items()
            if n > (1 if k == "sintesi" else MAX_NUMBERS_PER_SECTION)]
  virgola = bool(re.search(r"\d,\d", full))
  return {
      "numeri_citati": len(numeri), "numeri_non_verificati": non_verificati,
      "termini_vietati": vietati, "gergo_tecnico": gergo, "avvertenze_mancanti": mancanti,
      "parole_per_sezione": lunghezze, "limiti_superati": superati, "troppi_numeri": troppi,
      "virgola_decimale": virgola,
      "stile_ok": not (troppi or gergo or superati or virgola),
      "superata": not (non_verificati or vietati or mancanti),
  }
