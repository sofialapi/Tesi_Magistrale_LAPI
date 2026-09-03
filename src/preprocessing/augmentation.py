# processo di augmentation, sigmoid correction + trasformazioni geometriche e fotometriche
import torch
import torchvision.transforms.v2 as transforms
import numpy as np

class DermalImageAugmentor:
    def __init__(self, target_size=(224, 224), is_training=True):
        self.is_training = is_training
        self.target_size = target_size
        
        # 1. TRASFORMAZIONI DI BASE (Sia per Training che per Validation/Test)
        self.base_transforms = transforms.Compose([
            transforms.ToImage(),       # Converte l'array NumPy/PIL in un tensore PyTorch d'immagine
            transforms.ToDtype(torch.float32, scale=True), # Scalatura dei pixel in [0.0, 1.0]
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        ])
        
        # 2. TRASFORMAZIONI GEOMETRICHE E FOTOMETRICHE AVANZATE (Solo per Training)
        if self.is_training:
            self.augmentation_transforms = transforms.Compose([
                # Trasformazioni Geometriche 
                transforms.RandomRotation(degrees=(-30, 30), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.05, 0.05), # Leggero spostamento (shear/translation)
                    scale=(0.9, 1.1),       # Zoom del +-10%
                    shear=(-5, 5)           # Distorsione angolare (shear)
                ),
                # Trasformazioni Fotometriche 
                transforms.ColorJitter(
                    brightness=(0.8, 1.2),  # Regolazione luminosità
                    contrast=(0.8, 1.2)     # Regolazione contrasto
                )
            ])
            
    def _intensity_sigmoid_correction(self, img_tensor: torch.Tensor, cutoff=0.5, gain=10.0) -> torch.Tensor:
        """
        Applica la correzione del contrasto sigmoidea direttamente sul tensore PyTorch.
        Simula variazioni di illuminazione e fototipo senza alterare i pattern cromatici interni.
        Formula: 1 / (1 + exp(gain * (cutoff - x)))
        """
        # Applica la funzione sigmoide elemento per elemento sul tensore
        corrected_tensor = 1.0 / (1.0 + torch.exp(gain * (cutoff - img_tensor)))
        
        # Normalizzazione di sicurezza per mantenere i valori strettamente in [0.0, 1.0]
        corrected_tensor = (corrected_tensor - corrected_tensor.min()) / (corrected_tensor.max() - corrected_tensor.min() + 1e-6)
        return corrected_tensor

    def __call__(self, img_array: np.ndarray) -> torch.Tensor:
        """
        Rende la classe callable. Accetta un'immagine NumPy (letta da OpenCV)
        e restituisce il tensore pronto e aumentato per la rete neurale.
        """
        # Applica i passaggi di base (conversione a tensore, float32, resize)
        img_tensor = self.base_transforms(img_array)
        
        # Se siamo in fase di addestramento, applichiamo l'augmentation
        if self.is_training:
            img_tensor = self.augmentation_transforms(img_tensor)
            
            # Applica la Intensity Sigmoid Correction con il 30% di probabilità
            if torch.rand(1).item() < 0.3:
                img_tensor = self._intensity_sigmoid_correction(img_tensor)
                
        return img_tensor