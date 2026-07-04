import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import torch
from src.config import RAW_BCN_DIR, PROCESSED_BCN_DIR, RAW_ISIC_DIR, PROCESSED_ISIC_DIR
from src.preprocessing.image_preprocessing import process_image_pipeline
from src.preprocessing.dataset_manager import load_and_process_metadata, DermalMultimodalDataset
from src.preprocessing.augmentation import DermalImageAugmentor
import matplotlib.pyplot as plt

TARGET_IMAGE_SIZE = (224, 224)

def generate_comparison_plot(filename):
    """Genera un plot di controllo per verificare il ridimensionamento e il filtro gaussiano."""
    from src.config import PLOTS_DIR
    raw_path = os.path.join(RAW_ISIC_DIR, filename)
    processed_path = os.path.join(PROCESSED_ISIC_DIR, filename)
    
    img_raw = cv2.imread(raw_path)
    img_processed = cv2.imread(processed_path)
    
    img_raw_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_raw_rgb)
    axes[0].set_title(f"Originale (Grezza)\nDim: {img_raw.shape[:2]}")
    axes[0].axis('off')
    
    axes[1].imshow(img_processed_rgb)
    axes[1].set_title(f"Processata (Gauss + Resize)\nDim: {img_processed.shape[:2]}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"comparison_{os.path.splitext(filename)[0]}.png")
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Grafico comparativo salvato in: {plot_path}")

def generate_augmentation_plot(filename, raw_subdir):
    """Genera un grafico comparativo per validare visivamente l'Augmentation."""
    from src.config import PLOTS_DIR
    raw_path = os.path.join(raw_subdir, filename)
    img_bgr = cv2.imread(raw_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    augmentor = DermalImageAugmentor(target_size=TARGET_IMAGE_SIZE, is_training=True)
    aug_tensor_1 = augmentor(img_rgb)
    aug_tensor_2 = augmentor(img_rgb)
    
    aug_img_1 = aug_tensor_1.permute(1, 2, 0).numpy()
    aug_img_2 = aug_tensor_2.permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Immagine Originale (Raw)")
    axes[0].axis('off')
    
    axes[1].imshow(aug_img_1)
    axes[1].set_title("Augmentation Variante A")
    axes[1].axis('off')
    
    axes[2].imshow(aug_img_2)
    axes[2].set_title("Augmentation Variante B")
    axes[2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"aug_check_{os.path.splitext(filename)[0]}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[PLOT] Grafico di controllo dell'Augmentation salvato in: {plot_path}")

if __name__ == "__main__":
    print("=== Avvio Pipeline di Test Locale Multi-Dataset ===")
    
    # 1. Pipeline Immagini - Esecuzione su BCN
    print(f"\n[1/4] Elaborazione immagini BCN...")
    process_image_pipeline(RAW_BCN_DIR, PROCESSED_BCN_DIR, target_size=TARGET_IMAGE_SIZE)
    
    # 2. Pipeline Immagini - Esecuzione su ISIC
    print(f"\n[2/4] Elaborazione immagini ISIC...")
    process_image_pipeline(RAW_ISIC_DIR, PROCESSED_ISIC_DIR, target_size=TARGET_IMAGE_SIZE)
    
    # Controllo plot visivi su un campione reale ISIC
    processed_files = [f for f in os.listdir(PROCESSED_ISIC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if processed_files:
        generate_comparison_plot(processed_files[0])
        generate_augmentation_plot(processed_files[0], RAW_ISIC_DIR)
    
    print("-" * 50)
    print("[3/4] Avvio elaborazione metadati clinici strutturati (ISIC)...")
    
    # 3. Caricamento ed elaborazione dati clinici ISIC
    csv_full_path = os.path.join(RAW_ISIC_DIR, "metadata_sample.csv")
    X_clinical, y_isic, isic_ids = load_and_process_metadata(csv_full_path)
    
    print("-" * 50)
    print("[4/4] Test dei Dataset PyTorch nelle due modalità distinte...")
    
    # --- TEST MODALITÀ MULTIMODALE (Puntato su ISIC) ---
    print("\n>>> Creazione DermalMultimodalDataset in modalità [MULTIMODAL] (ISIC)...")
    isic_dataset = DermalMultimodalDataset(
        image_ids=isic_ids,
        labels=y_isic,
        image_dir=PROCESSED_ISIC_DIR,
        clinical_matrix=X_clinical,
        mode='multimodal',
        is_training=True
    )
    
    # Estraiamo il primo elemento del dataset multimodale per verificarne l'output
    img_t, clin_t, label_t = isic_dataset[0]
    print(f"[OK] Output Multimodal Campione 0 verificato:")
    print(f"  - Tensore Immagine Shape: {img_t.shape} (Atteso: [3, 224, 224])")
    print(f"  - Tensore Clinico Shape: {clin_t.shape} (Atteso: [10])")
    print(f"  - Target Label: {label_t.item()} (Atteso: 0 o 1)")
    
    # --- TEST MODALITÀ IMAGE-ONLY (Puntato su BCN) ---
    print("\n>>> Creazione DermalMultimodalDataset in modalità [IMAGE_ONLY] (BCN)...")
    
    # Ricaviamo gli ID delle immagini BCN elaborate salvate su disco
    bcn_processed_files = [os.path.splitext(f)[0] for f in os.listdir(PROCESSED_BCN_DIR) if f.lower().endswith('.jpg')]
    # Creiamo delle label finte per il test su BCN (es. alternanza di 0 e 1)
    y_bcn = [i % 2 for i in range(len(bcn_processed_files))]
    
    bcn_dataset = DermalMultimodalDataset(
        image_ids=bcn_processed_files,
        labels=y_bcn,
        image_dir=PROCESSED_BCN_DIR,
        clinical_matrix=None,  # Non passiamo metadati per la modalità image_only!
        mode='image_only',
        is_training=True
    )
    
    # Estraiamo il primo elemento del dataset image_only per verificarne l'output
    img_bcn_t, label_bcn_t = bcn_dataset[0]
    print(f"[OK] Output Image-Only Campione 0 verificato:")
    print(f"  - Tensore Immagine Shape: {img_bcn_t.shape} (Atteso: [3, 224, 224])")
    print(f"  - Target Label: {label_bcn_t.item()} (Atteso: 0 o 1)")
    print("  - Nota: Nessun vettore clinico restituito, coerenza preservata.")
    
    # --------------------------------------------------
    print("-" * 50)
    print("[DATALOADER STEP] Test dei DataLoader PyTorch...")
    from src.preprocessing.dataset_manager import create_multimodal_dataloader
    
    # Test DataLoader Multimodale (ISIC)
    print("\n>>> Generazione DataLoader per ISIC (Multimodal, batch_size=4)...")
    isic_loader = create_multimodal_dataloader(isic_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Estraiamo il primo batch per controllare le dimensioni aggregate
    for batch_idx, batch_data in enumerate(isic_loader):
        images_b, clinical_b, labels_b = batch_data
        print(f"[OK] Batch 0 di ISIC estratto con successo:")
        print(f"  - Tensore Immagini Batch Shape: {images_b.shape} (Atteso: [4, 3, 224, 224])")
        print(f"  - Tensore Clinico Batch Shape: {clinical_b.shape} (Atteso: [4, 10])")
        print(f"  - Tensore Target Batch Shape: {labels_b.shape} (Atteso: [4])")
        break # Ci basta controllare il primo batch per il debug
        
    # Test DataLoader Image-Only (BCN)
    print("\n>>> Generazione DataLoader per BCN (Image-Only, batch_size=4)...")
    bcn_loader = create_multimodal_dataloader(bcn_dataset, batch_size=4, shuffle=True, num_workers=2)
    
    for batch_idx, batch_data in enumerate(bcn_loader):
        images_bcn_b, labels_bcn_b = batch_data
        print(f"[OK] Batch 0 di BCN estratto con successo:")
        print(f"  - Tensore Immagini BCN Batch Shape: {images_bcn_b.shape} (Atteso: [4, 3, 224, 224])")
        print(f"  - Tensore Target BCN Batch Shape: {labels_bcn_b.shape} (Atteso: [4])")
        break

    print("\n=== Pipeline e Architettura Modale Completate con Successo ===")