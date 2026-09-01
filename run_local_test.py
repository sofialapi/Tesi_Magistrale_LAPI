import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2
import torch
import matplotlib.pyplot as plt

from src.config import RAW_BCN_DIR, PROCESSED_BCN_DIR, RAW_ISIC_DIR, PROCESSED_ISIC_DIR, PLOTS_DIR
from src.preprocessing.image_preprocessing import process_image_pipeline, dull_razor
from src.preprocessing.dataset_manager import load_and_process_metadata, DermalMultimodalDataset, create_multimodal_dataloader
from src.preprocessing.augmentation import DermalImageAugmentor

TARGET_IMAGE_SIZE = (224, 224)

def generate_comparison_plot(filename):
    """Genera un plot di controllo per verificare ridimensionamento e filtro Gaussiano."""
    raw_path = os.path.join(RAW_ISIC_DIR, filename)
    processed_path = os.path.join(PROCESSED_ISIC_DIR, filename)
    
    img_raw = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    img_processed = cv2.cvtColor(cv2.imread(processed_path), cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_raw)
    axes[0].set_title(f"Raw: {img_raw.shape[:2]}")
    axes[0].axis('off')
    
    axes[1].imshow(img_processed)
    axes[1].set_title(f"Processed: {img_processed.shape[:2]}")
    axes[1].axis('off')
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f"comparison_{os.path.splitext(filename)[0]}.png")
    plt.savefig(plot_path, dpi=100)
    plt.close()

def generate_augmentation_plot(filename, raw_subdir):
    """Genera un plot per verificare le trasformazioni di Data Augmentation."""
    raw_path = os.path.join(raw_subdir, filename)
    img_rgb = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    
    augmentor = DermalImageAugmentor(target_size=TARGET_IMAGE_SIZE, is_training=True)
    aug_1 = augmentor(img_rgb).permute(1, 2, 0).numpy()
    aug_2 = augmentor(img_rgb).permute(1, 2, 0).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Originale (Raw)")
    axes[0].axis('off')
    
    axes[1].imshow(aug_1)
    axes[1].set_title("Augmentation A")
    axes[1].axis('off')
    
    axes[2].imshow(aug_2)
    axes[2].set_title("Augmentation B")
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f"aug_check_{os.path.splitext(filename)[0]}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

#utilizzato solo per generare immagine comparativa per tesi, non richiamato in main
def generate_dull_razor_plot(filename, raw_subdir):
    """Genera un confronto isolato prima/dopo per l'algoritmo Dull Razor."""
    raw_path = os.path.join(raw_subdir, filename)
    img_raw = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    img_dull_razor = cv2.cvtColor(dull_razor(raw_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_raw)
    axes[0].set_title(f"Originale con Artefatti Piliferi\n{filename}")
    axes[0].axis('off')

    axes[1].imshow(img_dull_razor)
    axes[1].set_title(f"Post Dull Razor (Inpainting Telea)\n{filename}")
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f"dull_razor_check_{os.path.splitext(filename)[0]}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    print("=== Avvio Test Locale Pipeline ===")
    
    # Elaborazione Immagini BCN & ISIC
    print("\n[1/4] Pipeline Immagini BCN...")
    process_image_pipeline(RAW_BCN_DIR, PROCESSED_BCN_DIR, target_size=TARGET_IMAGE_SIZE)
    
    print("\n[2/4] Pipeline Immagini ISIC...")
    process_image_pipeline(RAW_ISIC_DIR, PROCESSED_ISIC_DIR, target_size=TARGET_IMAGE_SIZE)
    
    processed_files = [f for f in os.listdir(PROCESSED_ISIC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if processed_files:
        generate_comparison_plot(processed_files[0])
        generate_augmentation_plot(processed_files[0], RAW_ISIC_DIR)
        print("  -> Grafici di controllo generati in outputs/plots/")
    
    # Caricamento ed elaborazione dati clinici ISIC (con VIF)
    print("\n[3/4] Elaborazione Metadati Clinici (ISIC)...")
    csv_full_path = os.path.join(RAW_ISIC_DIR, "metadata_sample.csv")
    X_clinical, y_isic, isic_ids = load_and_process_metadata(csv_full_path)
    num_clinical_features = X_clinical.shape[1]
    
    # Test Dataset
    print("\n[4/4] Test Dataset PyTorch...")
    
    # Modalità Multimodale (ISIC)
    isic_dataset = DermalMultimodalDataset(
        image_ids=isic_ids,
        labels=y_isic,
        image_dir=PROCESSED_ISIC_DIR,
        clinical_matrix=X_clinical,
        mode='multimodal',
        is_training=True
    )
    img_t, clin_t, label_t = isic_dataset[0]
    print(f"  [OK] Campione Multimodal: Img {tuple(img_t.shape)}, Clin {tuple(clin_t.shape)} [Atteso: ({num_clinical_features},)], Label {label_t.item()}")
    
    # Modalità Image-Only (BCN)
    bcn_processed_files = [os.path.splitext(f)[0] for f in os.listdir(PROCESSED_BCN_DIR) if f.lower().endswith('.jpg')]
    y_bcn = [i % 2 for i in range(len(bcn_processed_files))]
    
    bcn_dataset = DermalMultimodalDataset(
        image_ids=bcn_processed_files,
        labels=y_bcn,
        image_dir=PROCESSED_BCN_DIR,
        clinical_matrix=None,
        mode='image_only',
        is_training=True
    )
    img_bcn_t, label_bcn_t = bcn_dataset[0]
    print(f"  [OK] Campione Image-Only: Img {tuple(img_bcn_t.shape)}, Label {label_bcn_t.item()}")
    
    # Test DataLoader
    print("\n[DataLoader Step] Test Batching...")
    batch_size_test = 4
    
    # Loader Multimodale
    isic_loader = create_multimodal_dataloader(isic_dataset, batch_size=batch_size_test, shuffle=True, num_workers=2)
    images_b, clinical_b, labels_b = next(iter(isic_loader))
    print(f"  [OK] Batch Multimodal: Img {tuple(images_b.shape)}, Clin {tuple(clinical_b.shape)}, Labels {tuple(labels_b.shape)}")
    
    # Loader Image-Only
    bcn_loader = create_multimodal_dataloader(bcn_dataset, batch_size=batch_size_test, shuffle=True, num_workers=2)
    images_bcn_b, labels_bcn_b = next(iter(bcn_loader))
    print(f"  [OK] Batch Image-Only: Img {tuple(images_bcn_b.shape)}, Labels {tuple(labels_bcn_b.shape)}")

    print("\n=== Pipeline e Architettura Modale Verificate con Successo ===")