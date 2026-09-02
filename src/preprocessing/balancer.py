import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from imblearn.over_sampling import SMOTE
from typing import Tuple

#GAN PER IMMAGINI
class SkinLesionGenerator(nn.Module):
    """
    Generatore DCGAN per la sintesi di immagini di lesioni cutanee (classe minoritaria).
    Mappa un vettore di rumore latente z in un tensore immagine (3, 224, 224).
    """
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Proiezione iniziale: latent_dim -> (feature_maps * 16) x 7 x 7
        self.init_proj = nn.Sequential(
            nn.Linear(latent_dim, feature_maps * 16 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 16 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Blocchi di convoluzione trasposta (upsampling: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224)
        self.upsample_blocks = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(feature_maps, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output normalizzato in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.init_proj(z)
        out = out.view(out.size(0), -1, 7, 7)
        return self.upsample_blocks(out)


class SkinLesionDiscriminator(nn.Module):
    """
    Discriminatore DCGAN per distinguere immagini reali da lesioni sintetiche.
    Input: tensore immagine (3, 224, 224) -> Output: probabilità scalare [0, 1].
    """
    def __init__(self, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 1x1
            nn.Conv2d(feature_maps * 16, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img).view(-1, 1)


class DermalGANBalancer:
    """
    Interfaccia di alto livello per il modulo generativo di bilanciamento immagini.
    """
    def __init__(self, latent_dim: int = 100, device: str = None):
        self.latent_dim = latent_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = SkinLesionGenerator(latent_dim=latent_dim).to(self.device)
        self.discriminator = SkinLesionDiscriminator().to(self.device)

    def generate_synthetic_lesions(self, num_samples: int) -> torch.Tensor:
        """Genera un batch di immagini sintetiche della classe minoritaria."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            synthetic_imgs = self.generator(z)
            # Riporta da range Tanh [-1, 1] al range standard [0, 1]
            synthetic_imgs = (synthetic_imgs + 1.0) / 2.0
        return synthetic_imgs

    # Spazio riservato per training loop GAN su server Tesla

# SMOTE PER DATI TABULARI
class TabularSMOTEBalancer:
    """
    Modulo di sovracampionamento sintetico (SMOTE) per feature cliniche tabulari.
    Interpola linearmente nello spazio continuo dei k-vicini della classe minoritaria.
    """
    def __init__(self, random_state: int = 42, k_neighbors: int = 5):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applica SMOTE garantendo che il numero di k-neighbors sia compatibile
        con la numerosità della classe minoritaria (utile per mini-batch di test).
        """
        classes, counts = np.unique(y, return_counts=True)
        min_samples = np.min(counts)
        
        # Protezione per set di dati ridotti (es. test locali o fold piccoli)
        effective_k = min(self.k_neighbors, max(1, min_samples - 1))
        
        smote = SMOTE(k_neighbors=effective_k, random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled