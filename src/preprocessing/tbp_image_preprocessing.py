"""tbp_image_preprocessing.py

Modulo di pre-elaborazione dedicato per crop di lesioni cutanee estratti da 3D
Total Body Photography (ISIC 2024).
Supera i limiti di DullRazor introducendo Color Constancy, CLAHE adattivo su
spazio LAB e filtraggio artefatti conservativo.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


def apply_color_constancy_shades_of_gray(
    img: np.ndarray, power: int = 6
) -> np.ndarray:
    """Standardizza l'illuminazione dell'immagine TBP stimando l'illuminante globale

    tramite l'algoritmo di Minkowski p-norm (Shades of Gray).
    """
    img_float = img.astype(np.float32) + 1e-5
    # Calcolo dell'illuminante per ciascun canale RGB
    illuminant = (np.mean(img_float**power, axis=(0, 1))) ** (1.0 / power)
    illuminant_norm = np.sqrt(np.sum(illuminant**2)) + 1e-5
    illuminant /= illuminant_norm

    # Correzione del bilanciamento del bianco
    img_normalized = img_float / (illuminant * np.sqrt(3))
    return np.clip(img_normalized, 0, 255).astype(np.uint8)


def enhance_contrast_clahe(
    img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)
) -> np.ndarray:
    """Esalta i dettagli della lesione senza amplificare il rumore di fondo

    applicando CLAHE sul solo canale L (luminanza) nello spazio colore CIELAB.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size
    )
    l_enhanced = clahe.apply(l_channel)

    lab_merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)


def soft_hair_attenuation(img: np.ndarray) -> np.ndarray:
    """Attenua artefatti lineari scuri senza corrompere i gradienti interni della lesione,

    evitando l'inpainting distruttivo di DullRazor standard.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kernel allungati multidirezionali per isolare elementi lineari fini
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))

    bh_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_horiz)
    bh_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_vert)
    hair_map = cv2.max(bh_h, bh_v)

    # Soglia conservativa: isola solo elementi con gradiente molto netto
    _, mask = cv2.threshold(hair_map, 25, 255, cv2.THRESH_BINARY)

    # Inpainting a raggio minimo (1px) per non sfumare il pattern pigmentario
    if np.count_nonzero(mask) > 0:
        return cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return img


def process_tbp_image(
    image_path: str, target_size=(224, 224)
) -> np.ndarray:
    """Pipeline integrata per singola immagine 3D TBP."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"Impossibile leggere l'immagine al percorso: {image_path}"
        )

    # 1. Normalizzazione cromatica (correzione ombre e luci da fotocamere 3D)
    img_cc = apply_color_constancy_shades_of_gray(img, power=6)

    # 2. Filtraggio conservativo artefatti lineari
    img_clean = soft_hair_attenuation(img_cc)

    # 3. Esaltazione del contrasto locale su canale L
    img_enhanced = enhance_contrast_clahe(img_clean, clip_limit=1.5)

    # 4. Ridimensionamento con interpolazione per preservare i pixel fini
    h, w = img.shape[:2]
    interp = (
        cv2.INTER_AREA
        if (h > target_size[1] and w > target_size[0])
        else cv2.INTER_LINEAR
    )
    img_resized = cv2.resize(img_enhanced, target_size, interpolation=interp)

    return img_resized


def process_tbp_pipeline(
    input_dir: str, output_dir: str, target_size=(224, 224)
):
    """Elabora in batch una cartella di crop TBP e salva i risultati preservando i nomi file."""
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    if not files:
        print(f"[ATTENZIONE] Nessuna immagine valida trovata in '{input_dir}'")
        return

    print(
        f"Inizio elaborazione TBP avanzata su {len(files)} campioni da '{input_dir}'..."
    )
    for filename in tqdm(files):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        try:
            processed = process_tbp_image(in_path, target_size=target_size)
            cv2.imwrite(out_path, processed)
        except Exception as e:
            print(f"Errore su {filename}: {e}")