import os
import sys

# Determina la cartella radice (Tesi_Magistrale_LAPI) in modo assoluto
root_dir = os.path.dirname(os.path.abspath(__file__))

# Inserita in posizione 0 (priorità massima) nel path di Python
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import cv2
import matplotlib.pyplot as plt
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, TARGET_IMAGE_SIZE
from src.preprocessing.image_preprocessing import process_image_pipeline
from src.preprocessing.dataset_manager import load_and_process_metadata

def generate_comparison_plot(image_name: str):
    """
    Genera e salva un grafico di confronto Prima/Dopo per una specifica immagine.
    """
    raw_path = os.path.join(RAW_DATA_DIR, image_name)
    processed_path = os.path.join(PROCESSED_DATA_DIR, image_name)
    
    # Controlla se entrambi i file esistono prima di procedere
    if not os.path.exists(raw_path) or not os.path.exists(processed_path):
        print(f"Impossibile generare il plot comparativo per {image_name}: file mancanti.")
        return

    # Carica le immagini (OpenCV legge in BGR, convertiamo in RGB per Matplotlib)
    img_raw = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    img_processed = cv2.cvtColor(cv2.imread(processed_path), cv2.COLOR_BGR2RGB)

    # Crea la figura con i due subplot affiancati
    plt.figure(figsize=(10, 5))
    
    # Subplot Immagine Originale
    plt.subplot(1, 2, 1)
    plt.imshow(img_raw)
    plt.title(f"Originale ({img_raw.shape[1]}x{img_raw.shape[0]})")
    plt.axis('off')
    
    # Subplot Immagine Elaborata
    plt.subplot(1, 2, 2)
    plt.imshow(img_processed)
    plt.title(f"Processed ({TARGET_IMAGE_SIZE[0]}x{TARGET_IMAGE_SIZE[1]})")
    plt.axis('off')
    
    # Salva il plot del confronto visivo
    plot_output_path = os.path.join(PLOTS_DIR, f"comparison_{image_name.split('.')[0]}.png")
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()
    print(f"Grafico comparativo salvato in: {plot_output_path}")

if __name__ == "__main__":
    from src.config import RAW_BCN_DIR, PROCESSED_BCN_DIR, RAW_ISIC_DIR, PROCESSED_ISIC_DIR
    
    print("=== Avvio Pipeline di Test Locale Multi-Dataset ===")
    
    # 1. Pipeline Immagini - Esecuzione su BCN
    print(f"\n[1/3] Elaborazione immagini BCN...")
    process_image_pipeline(RAW_BCN_DIR, PROCESSED_BCN_DIR, target_size=TARGET_IMAGE_SIZE)
    
    # 2. Pipeline Immagini - Esecuzione su ISIC (Aggiunto!)
    print(f"\n[2/3] Elaborazione immagini ISIC...")
    process_image_pipeline(RAW_ISIC_DIR, PROCESSED_ISIC_DIR, target_size=TARGET_IMAGE_SIZE)
    
    # Generazione plot di controllo visivo per la prima immagine di ISIC
    processed_files = [f for f in os.listdir(PROCESSED_ISIC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if processed_files:
        RAW_DATA_DIR = RAW_ISIC_DIR
        PROCESSED_DATA_DIR = PROCESSED_ISIC_DIR
        generate_comparison_plot(processed_files[0])
    
    print("-" * 50)
    print("[3/3] Avvio elaborazione metadati clinici strutturati (ISIC)...")
    
    # 3. Pipeline Tabulare (Punta al CSV di ISIC)
    try:
        csv_full_path = os.path.join(RAW_ISIC_DIR, "metadata_sample.csv")
        X_clinical, y, ids = load_and_process_metadata(csv_full_path)
        print(f"Matrice delle feature cliniche generata! Forma: {X_clinical.shape}")
        print(f"Primo paziente codificato (Z-score + One-Hot):\n{X_clinical[0]}")
    except Exception as e:
        print(f"[ERRORE TABULARE] Qualcosa è andato storto: {str(e)}")
        
    print("\n=== Pipeline Completata con Successo ===")