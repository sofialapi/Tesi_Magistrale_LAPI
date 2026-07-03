#implementazione degli algoritmi di pulizia delle immagini
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def dull_razor(image_path: str) -> np.ndarray:
    """
    Applica l'algoritmo Dull Razor per rimuovere i peli dalle immagini dermatologiche.
    
    Args:
        image_path (str): Path dell'immagine di input.
        
    Returns:
        np.ndarray: Immagine pulita in formato OpenCV (BGR).
    """
    # 1. Carica l'immagine a colori e convertila in scala di grigi
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Applica il filtro BlackHat per evidenziare gli elementi scuri e lineari (i peli)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Binarizzazione della mappa dei peli (creazione della maschera)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Inpainting: sostituisce i pixel dei peli interpolando con i pixel circostanti
    clean_img = cv2.inpaint(img, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    
    return clean_img

def standardize_image(img_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Ridimensiona l'immagine e applica un filtro Gaussiano leggero per ridurre il rumore.
    """
    # Ridimensionamento bilineare (ottimale per CNN/ViT)
    resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Riduzione del rumore ad alta frequenza
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    
    return denoised

import os
from tqdm import tqdm

def process_image_pipeline(input_dir: str, output_dir: str, target_size=(224, 224)):
    """
    Prende tutte le immagini da una cartella raw, applica Dull Razor,
    le standardizza e le salva nella cartella processed.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Elenca i file immagine supportati
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    # DIAGNOSTICA: Se la lista è vuota, stampiamo un avviso chiaro per capire cosa c'è nella cartella
    if len(image_files) == 0:
        print(f"\n[ATTENZIONE] Nessun file immagine valido trovato in '{input_dir}'!")
        print(f"Contenuto attuale della cartella: {os.listdir(input_dir)}\n")
        return

    print(f"Trovate {len(image_files)} immagini in {input_dir}. Inizio elaborazione...")
    
    for filename in tqdm(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # 1. Rimuovi i peli (Dull Razor)
            clean_img = dull_razor(input_path)
            
            # 2. Ridimensiona e denoise
            final_img = standardize_image(clean_img, target_size=target_size)
            
            # 3. Salva l'immagine elaborata
            cv2.imwrite(output_path, final_img)
            
        except Exception as e:
            print(f"Errore durante l'elaborazione di {filename}: {str(e)}")