"""Verifica automatica di fedelta' della spiegazione rispetto ai descrittori.

Controlli:
- numeri citati nel testo che non trovano corrispondenza nei descrittori;
- avvertenze calcolate a regole non riportate dal testo;
- termini dermoscopici vietati (strutture non misurate);
- superamento dei limiti di parole.
"""
import re

from .nl_config import WORD_LIMITS

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
  for v in allowed:
    if abs(x - v) <= max(0.006, 0.01 * abs(v)):
      return True
    if abs(x - 100 * v) <= 0.6 or abs(x / 100 - v) <= 0.006:  # % vs frazione
      return True
  return False


def _words(s):
  return len(re.findall(r"\w+", s))


def verify_explanation(sections, features, rule_warns):
  allowed = []
  _collect_numbers(features, allowed)
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

  return {
      "numeri_citati": len(numeri),
      "numeri_non_verificati": non_verificati,
      "termini_vietati": vietati,
      "avvertenze_mancanti": mancanti,
      "parole_per_sezione": lunghezze,
      "limiti_superati": superati,
      "superata": not (non_verificati or vietati or mancanti),
  }
