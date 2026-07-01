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
    print("=== Avvio Pipeline di Test Locale ===")
    print(f"Directory Raw: {RAW_DATA_DIR}")
    print(f"Directory Processed: {PROCESSED_DATA_DIR}")
    print("-" * 40)

    # 1. Esegui la pipeline di elaborazione su tutte le 10 immagini
    process_image_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR, target_size=TARGET_IMAGE_SIZE)
    
    # 2. Genera un grafico di controllo visivo per la prima immagine elaborata
    processed_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if processed_files:
        print("-" * 40)
        print("Generazione dei grafici di controllo visivo...")
        generate_comparison_plot(processed_files[0])
    
    print("\n=== Test Completato con Successo ===")