"""Verifica automatica di fedelta' della spiegazione rispetto ai descrittori.

Controlli:
- numeri citati nel testo che non trovano corrispondenza nei descrittori;
- avvertenze calcolate a regole non riportate dal testo;
- termini dermoscopici vietati (strutture non misurate);
- superamento dei limiti di parole.
Controlli di stile (non bloccanti): troppi numeri per sezione, gergo tecnico.

I numeri sono confrontati con i soli descrittori inviati all'LLM (llm_view).
"""
import re

from .nl_config import WORD_LIMITS
from .nl_prompts import llm_view

MAX_NUMBERS_PER_SECTION = 2
_JARGON = {
    "IoU": r"\biou\b",
    "CIELAB": r"cielab|[\u0394\u03b4]\s?[lab]\b|delta[_ ]?[lab]\b",
    "energia": r"\benergi\w*",
    "regione/area calda": r"\b(regione|area|zona)\s+cald\w*",
    "correlazione": r"\bcorrelazion\w*",
    "copertura": r"\bcopertur\w*",
    "pixel": r"\bpixel\b",
}

# Pattern che contengono cifre ma non sono valori da verificare
_IGNORE = re.compile(
    r"ResNet-?\s?50|MobileViT-?\s?S|Grad-?CAM\+*|GPT-OSS-?20B|ISIC_\d+|"
    r"\d+\s*[x×]\s*\d+|\bL\*|\ba\*|\bb\*|top-?\s?20\s?%?",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")

_FORBIDDEN = {
    "reticolo": r"\breticol\w*",
    "strie": r"\bstri[ea]\b",
    "globuli": r"\bglobul\w*",
    "velo": r"\bvelo\b",
    "regressione": r"\bregression\w*",
    "pseudopodi": r"\bpseudopod\w*",
    "vasi": r"\bvas[oi]\b|\bvascolar\w*",
    "diagnosi": r"\bdiagnosi\s+di\b|\bsi\s+tratta\s+di\s+(?:un\s+)?melanoma\b",
}

# Parole chiave con cui un'avvertenza a regole si considera riportata
_WARN_KEYS = {
    "SEGMENTAZIONE_INCERTA": r"segmentaz|delimitaz|contorno|incert",
    "FUORI_LESIONE": r"fuori|estern|esterior|\bcute\b|perilesion|non\s+clinicamente",
    "PICCO_FUORI_LESIONE": r"fuori|estern|\bcute\b|perilesion",
    "MARGINE_IMMAGINE": r"bordo\s+dell'?\s?immagine|margine\s+dell'?\s?immagine|angolo|artefatt",
    "VICINO_SOGLIA": r"soglia",
    "DISACCORDO": r"disaccord|discord|divers|diverg|contrastant",
}


def _collect_numbers(obj, out):
  if isinstance(obj, bool) or obj is None:
    return
  if isinstance(obj, (int, float)):
    out.append(float(obj))
  elif isinstance(obj, dict):
    for v in obj.values():
      _collect_numbers(v, out)
  elif isinstance(obj, (list, tuple)):
    for v in obj:
      _collect_numbers(v, out)
  elif isinstance(obj, str):
    for m in _NUM.findall(_IGNORE.sub(" ", obj)):
      out.append(float(m.replace(",", ".")))


def _matches(x, allowed):
  """x (numero citato) corrisponde a un descrittore, a meno del segno.

  Il segno viene ignorato perche' il testo puo' usare il meno tipografico.
  La conversione percentuale/frazione vale solo nei casi sensati
  (90.7 <-> 0.907, 0.24 <-> 24), non per numeri piccoli contro zero.
  """
  ax = abs(x)
  for v in allowed:
    av = abs(v)
    if abs(ax - av) <= max(0.006, 0.01 * av):
      return True
    if ax >= 1 and av < 1 and abs(ax - 100 * av) <= 0.6:
      return True
    if ax < 1 and av >= 1 and abs(100 * ax - av) <= 0.6:
      return True
  return False


def _words(s):
  return len(re.findall(r"\w+", s))


def verify_explanation(sections, features, rule_warns):
  allowed = []
  _collect_numbers(llm_view(features), allowed)
  allowed += [224.0]

  full_text = " ".join([sections[k] for k in sections if k != "avvertenze"]
                       + list(sections["avvertenze"]))
  clean = _IGNORE.sub(" ", full_text)
  numeri = [float(n.replace(",", ".")) for n in _NUM.findall(clean)]
  non_verificati = sorted({n for n in numeri if not _matches(n, allowed)})

  low = full_text.lower()
  vietati = [name for name, pat in _FORBIDDEN.items() if re.search(pat, low)]

  avv_text = " ".join(sections["avvertenze"]).lower()
  mancanti = []
  for w in rule_warns:
    pat = _WARN_KEYS.get(w["codice"], r"$^")
    # per le avvertenze di un singolo modello si richiede anche il nome
    name_ok = True
    if w["modello"] == "resnet50":
      name_ok = "resnet" in avv_text
    elif w["modello"] == "mobilevit_s":
      name_ok = "mobilevit" in avv_text
    if not (re.search(pat, avv_text) and name_ok):
      mancanti.append(w["codice"] + (f"@{w['modello']}" if w["modello"] else ""))

  lunghezze = {k: _words(sections[k]) for k in sections if k != "avvertenze"}
  superati = [k for k, n in lunghezze.items()
              if n > WORD_LIMITS.get(k, 10**6) * 1.2]
  superati += [f"avvertenza_{i+1}" for i, a in enumerate(sections["avvertenze"])
               if _words(a) > WORD_LIMITS["avvertenza"] * 1.2]

  # --- stile (non bloccante) ---
  numeri_per_sezione = {
      k: len(_NUM.findall(_IGNORE.sub(" ", sections[k])))
      for k in sections if k != "avvertenze"}
  troppi_numeri = [k for k, n in numeri_per_sezione.items()
                   if n > MAX_NUMBERS_PER_SECTION]
  gergo = [name for name, pat in _JARGON.items() if re.search(pat, low)]

  return {
      "numeri_citati": len(numeri),
      "numeri_per_sezione": numeri_per_sezione,
      "troppi_numeri": troppi_numeri,
      "gergo_tecnico": gergo,
      "stile_ok": not (troppi_numeri or gergo or superati),
      "numeri_non_verificati": non_verificati,
      "termini_vietati": vietati,
      "avvertenze_mancanti": mancanti,
      "parole_per_sezione": lunghezze,
      "limiti_superati": superati,
      "superata": not (non_verificati or vietati or mancanti),
  }
