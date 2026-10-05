"""XAI_NL: spiegazioni in linguaggio naturale delle mappe Grad-CAM++.

Pipeline: immagine preelaborata -> predizione + Grad-CAM++ -> descrittori
quantitativi (nl_cam_features) -> testo generato da GPT-OSS-20B via Groq
(nl_explainer) -> verifica di fedelta' (nl_verify) -> figura con testo (nl_figure).

Il pacchetto non importa torch a livello di modulo: nl_models viene importato
solo dallo script di esecuzione, cosi' selftest e figure funzionano anche senza GPU.
"""
