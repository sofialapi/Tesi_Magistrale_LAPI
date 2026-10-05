# XAI_NL: spiegazioni in linguaggio naturale delle mappe Grad-CAM++

Modulo per BCN20000 e HAM10000 (modelli image-only addestrati su immagini preelaborate).
Per ogni caso produce una figura con immagine, mappe di ResNet-50 e MobileViT-S e,
sotto, una spiegazione scritta per il clinico generata da GPT-OSS-20B via Groq.

## File

| File | Ruolo |
|---|---|
| `nl_config.py` | percorsi, modello LLM, soglie dei descrittori, limiti di parole |
| `nl_models.py` | caricamento checkpoint, input preelaborato, Grad-CAM++ |
| `nl_cam_features.py` | segmentazione euristica, descrittori quantitativi, avvertenze a regole |
| `nl_prompts.py` | prompt di sistema e schema JSON della risposta |
| `nl_explainer.py` | `GroqExplainer` (LLM, con cache) e `TemplateExplainer` (baseline) |
| `nl_verify.py` | verifica di fedeltà: numeri, termini vietati, avvertenze, lunghezza |
| `nl_figure.py` | figura con testo sotto ciascun pannello |
| `nl_run_explain.py` | script principale (BCN e HAM) |
| `nl_selftest.py` | test offline senza GPU né API |

## Installazione (Tesla)

Copia la cartella `XAI_NL/` nella root del progetto, poi:

```bash
pip install groq            # gli altri pacchetti sono già nell'ambiente tesi_lapi
export GROQ_API_KEY="..."   # oppure nel ~/.bashrc
python -m XAI_NL.nl_selftest            # controllo offline della figura
```

## Uso

Sempre dalla root del progetto. Prima di tutto controlla in `nl_config.py`
che `csv`, `processed_dir`, `raw_dir` e i checkpoint corrispondano ai percorsi sulla Tesla.

```bash
# casi delle figure 6.1 e 6.2 della tesi
python -m XAI_NL.nl_run_explain --dataset bcn --ids ISIC_0000002
python -m XAI_NL.nl_run_explain --dataset ham --ids ISIC_0028086

# 3 melanomi + 3 benigni casuali dal fold di validazione del checkpoint
python -m XAI_NL.nl_run_explain --dataset ham --fold 1 --num_samples 6

# stesso run senza API (testo da template)
python -m XAI_NL.nl_run_explain --dataset ham --fold 1 --no_llm
```

Opzioni utili: `--threshold`, `--reasoning_effort low|medium|high`,
`--overlay_on raw` (mappe sovrapposte all'immagine originale), `--no_contour`,
`--selection first`, `--seed`.

## Output

- `figure/xai_nl/{bcn,ham}/{ds}_nl_gradcam_NN_<ID>_<GT>.png`: figura per la tesi
- `outputs/xai_nl/{bcn,ham}/...json`: descrittori inviati, risposta, verifica
- `outputs/xai_nl/{bcn,ham}/{ds}_nl_summary.csv`: riepilogo per tabelle
- `outputs/xai_nl/cache/`: risposte LLM (stesso caso e prompt = nessuna nuova chiamata).
  Cambia `PROMPT_VERSION` quando modifichi il prompt.

## Scelte metodologiche (da riportare nella tesi)

- **Input coerente con il training**: i modelli ricevono l'immagine di `data/processed`
  (DullRazor + filtro gaussiano). Il primo pannello mostra l'originale al clinico;
  le mappe sono sovrapposte all'immagine vista dal modello. DullRazor non altera la
  geometria, quindi le due immagini sono allineate.
- **LLM senza accesso all'immagine**: GPT-OSS-20B riceve solo descrittori numerici e
  categoriali. Ogni affermazione è riconducibile a un valore nel JSON del caso.
- **Ground truth esclusa** dal testo inviato all'LLM.
- **Avvertenze a regole**: calcolate in modo deterministico; se l'LLM ne omette una,
  viene aggiunta in figura con la dicitura "(aggiunta automaticamente)".
- **Segmentazione euristica** (Otsu su L*): il contorno è mostrato in figura; se
  implausibile, viene segnalata un'avvertenza.
- **Fold**: per HAM usa il fold di validazione del checkpoint. Per BCN il CSV non ha
  una colonna `fold`: usa `--ids` di casi di validazione o aggiungi la colonna.
