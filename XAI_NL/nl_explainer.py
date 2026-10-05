"""Generazione della spiegazione: GPT-OSS-20B via Groq, oppure template a regole.

Il TemplateExplainer serve per i test senza API e come baseline da confrontare
con l'LLM nella tesi (stessi descrittori, testo deterministico).
"""
import hashlib
import json
import os
import re
import time

from .nl_config import LLM_MODEL, MODEL_KEYS, PROMPT_VERSION
from .nl_prompts import (RESPONSE_SCHEMA, SECTION_KEYS, SYSTEM_PROMPT,
                         build_user_prompt)


def _parse_sections(content):
  text = (content or "").strip()
  text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
  data = json.loads(text)
  for k in SECTION_KEYS:
    if k not in data:
      raise ValueError(f"sezione mancante: {k}")
  if isinstance(data["avvertenze"], str):
    data["avvertenze"] = [data["avvertenze"]] if data["avvertenze"].strip() else []
  return {k: data[k] for k in SECTION_KEYS}


class GroqExplainer:
  name = "llm"

  def __init__(self, model=LLM_MODEL, reasoning_effort="medium", temperature=0.3,
               max_completion_tokens=4096, cache_dir=None, max_retries=3):
    try:
      from groq import Groq
    except ImportError as e:
      raise ImportError("Client Groq mancante: pip install groq") from e
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
      raise RuntimeError("Imposta la variabile d'ambiente GROQ_API_KEY "
                         "(oppure usa --no_llm per il template).")
    self.client = Groq(api_key=api_key)
    self.model = model
    self.reasoning_effort = reasoning_effort
    self.temperature = temperature
    self.max_completion_tokens = max_completion_tokens
    self.max_retries = max_retries
    self.cache_dir = cache_dir
    if cache_dir:
      os.makedirs(cache_dir, exist_ok=True)
    # Varianti in ordine di preferenza; si scende se l'API rifiuta un parametro.
    self._variants = [
        ("json_schema", {
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "spiegazione_xai", "strict": True,
                "schema": RESPONSE_SCHEMA}},
            "reasoning_effort": reasoning_effort,
            "include_reasoning": False}),
        ("json_object", {"response_format": {"type": "json_object"},
                         "reasoning_effort": reasoning_effort}),
        ("json_object_base", {"response_format": {"type": "json_object"}}),
    ]
    self._variant_idx = 0

  def _cache_path(self, features):
    blob = json.dumps({"v": PROMPT_VERSION, "m": self.model,
                       "e": self.reasoning_effort, "t": self.temperature,
                       "f": features}, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]
    return os.path.join(self.cache_dir, f"{h}.json") if self.cache_dir else None

  def explain(self, features):
    cache_path = self._cache_path(features)
    if cache_path and os.path.exists(cache_path):
      with open(cache_path, encoding="utf-8") as f:
        cached = json.load(f)
      cached["meta"]["cache"] = True
      return cached

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(features)}]
    last_err, attempts = None, 0
    while attempts < self.max_retries and self._variant_idx < len(self._variants):
      vname, extra = self._variants[self._variant_idx]
      t0 = time.time()
      try:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens, **extra)
      except TypeError as e:  # parametro non supportato dal client installato
        last_err = e
        self._variant_idx += 1
        continue
      except Exception as e:  # noqa: BLE001
        last_err = e
        status = getattr(e, "status_code", None)
        if status == 400:  # parametro rifiutato dall'API: variante successiva
          self._variant_idx += 1
          continue
        attempts += 1
        time.sleep(2 ** attempts)  # rate limit / errori temporanei
        continue

      latency = time.time() - t0
      try:
        sections = _parse_sections(resp.choices[0].message.content)
      except (ValueError, json.JSONDecodeError) as e:
        last_err = e
        attempts += 1
        continue

      usage = getattr(resp, "usage", None)
      result = {"sezioni": sections, "meta": {
          "explainer": "llm", "modello": self.model,
          "prompt_version": PROMPT_VERSION,
          "reasoning_effort": self.reasoning_effort,
          "temperature": self.temperature, "response_format": vname,
          "latenza_s": round(latency, 2), "cache": False,
          "token": {
              "prompt": getattr(usage, "prompt_tokens", None),
              "completion": getattr(usage, "completion_tokens", None),
              "totale": getattr(usage, "total_tokens", None)}}}
      if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
          json.dump(result, f, ensure_ascii=False, indent=1)
      return result
    raise RuntimeError(f"Generazione LLM fallita: {last_err!r}")


class TemplateExplainer:
  """Testo deterministico costruito dai descrittori (baseline e test offline)."""
  name = "template"

  @staticmethod
  def _model_text(m):
    mp = m["mappa"]
    if mp.get("energia_dentro_lesione_pct") is None:
      return (f"La mappa occupa il {mp['area_calda_pct_immagine']}% dell'immagine; "
              "la lesione non e' stata delimitata, quindi la posizione rispetto "
              "alla lesione non e' valutabile.")
    prep = {"centro della lesione": "nel", "zona intermedia della lesione": "nella",
            "margine della lesione": "sul", "cute perilesionale": "sulla",
            "cute distante dalla lesione": "sulla"}
    where = f"{prep.get(mp['zona_picco'], 'in')} {mp['zona_picco']}"
    if mp["direzione_picco"] not in (None, "centrale"):
      where += f", porzione {mp['direzione_picco']}"
    txt = (f"La massima attivazione si trova {where}. "
           f"Il {mp['energia_dentro_lesione_pct']}% dell'attivazione cade sulla "
           f"lesione; la mappa e' {mp['distribuzione']}")
    if mp.get("attivazione_margini_vs_centro"):
      txt += f", {mp['attivazione_margini_vs_centro']}"
    txt += "."
    if mp.get("regione_calda_rispetto_al_resto"):
      txt += (" La regione evidenziata appare "
              + ", ".join(mp["regione_calda_rispetto_al_resto"])
              + " rispetto al resto della lesione.")
    return txt

  def explain(self, features):
    t0 = time.time()
    mods = features["modelli"]
    parts = []
    for k in MODEL_KEYS:
      p = mods[k]["predizione"]
      parts.append(f"{mods[k]['nome']} stima P(melanoma) {p['probabilita_melanoma']} "
                   f"(esito {p['esito']}, sicurezza {p['sicurezza']})")
    sintesi = "; ".join(parts) + "."
    c = features["confronto_modelli"]
    confronto = (f"Le regioni piu' attive dei due modelli hanno sovrapposizione "
                 f"{c['sovrapposizione']}; gli esiti "
                 f"{'coincidono' if c['stesso_esito'] else 'sono diversi'}.")
    sections = {
        "sintesi": sintesi,
        "resnet50": self._model_text(mods["resnet50"]),
        "mobilevit_s": self._model_text(mods["mobilevit_s"]),
        "confronto": confronto,
        "avvertenze": list(features.get("avvertenze_calcolate", [])),
    }
    return {"sezioni": sections, "meta": {
        "explainer": "template", "prompt_version": PROMPT_VERSION,
        "latenza_s": round(time.time() - t0, 4), "cache": False}}
