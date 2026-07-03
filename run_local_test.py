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

from src.preprocessing.augmentation import DermalImageAugmentor

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

def generate_augmentation_plot(filename, raw_subdir, processed_subdir):
    """
    Genera un grafico comparativo per validare visivamente l'Augmentation e la Sigmoid Correction.
    """
    from src.config import PLOTS_DIR
    import cv2
    
    raw_path = os.path.join(raw_subdir, filename)
    img_bgr = cv2.imread(raw_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Inizializza l'augmentor in modalità training
    augmentor = DermalImageAugmentor(target_size=(224, 224), is_training=True)
    
    # Generiamo due varianti aumentate diverse della stessa immagine di partenza
    aug_tensor_1 = augmentor(img_rgb)
    aug_tensor_2 = augmentor(img_rgb)
    
    # Riconvertiamo i tensori di PyTorch in array NumPy per Matplotlib [C, H, W] -> [H, W, C]
    aug_img_1 = aug_tensor_1.permute(1, 2, 0).numpy()
    aug_img_2 = aug_tensor_2.permute(1, 2, 0).numpy()
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Immagine Originale (Raw)")
    axes[0].axis('off')
    
    axes[1].imshow(aug_img_1)
    axes[1].set_title("Augmentation Variante A\n(Geom + Foto)")
    axes[1].axis('off')
    
    axes[2].imshow(aug_img_2)
    axes[2].set_title("Augmentation Variante B\n(Geom + Foto + Sigmoid)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"aug_check_{os.path.splitext(filename)[0]}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[PLOT] Grafico di controllo dell'Augmentation salvato in: {plot_path}")

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
        # Genera il grafico di controllo per l'Augmentation
        generate_augmentation_plot(processed_files[0], RAW_ISIC_DIR, PROCESSED_ISIC_DIR)
    
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