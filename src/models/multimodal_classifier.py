import torch
import torch.nn as nn
from typing import Optional
from src.models.backbones import ResNetBackbone, HybridVisionBackbone, TabularMLP

class DermalClassifier(nn.Module):
    """
    Architettura unificata per i 4 casi di studio:
    - Caso 1: CNN Pura (Image-Only)
    - Caso 2: CNN+ViT Ibrido (Image-Only)
    - Caso 3: CNN Pura + Late Fusion (Multimodal)
    - Caso 4: CNN+ViT Ibrido + Late Fusion (Multimodal)
    """
    def __init__(
        self,
        backbone_type: str = 'cnn',          # 'cnn' oppure 'hybrid'
        mode: str = 'multimodal',             # 'image_only' oppure 'multimodal'
        num_classes: int = 2,                 # Classificazione binaria (0/1) o multiclasse
        clinical_dim: int = 9,                # Dimensione feature tabulari post-VIF
        clinical_embed_dim: int = 64,
        dropout_rate: float = 0.3,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        self.mode = mode.lower()
        
        # 1. Selezione Backbone Visivo
        if self.backbone_type == 'cnn':
            self.visual_backbone = ResNetBackbone(pretrained=pretrained)
        elif self.backbone_type == 'hybrid':
            self.visual_backbone = HybridVisionBackbone(pretrained=pretrained)
        else:
            raise ValueError("backbone_type deve essere 'cnn' o 'hybrid'")
            
        vis_dim = self.visual_backbone.feature_dim
        
        # 2. Modulo tabulare (se multimodal)
        if self.mode == 'multimodal':
            self.tabular_encoder = TabularMLP(
                in_features=clinical_dim, 
                embedding_dim=clinical_embed_dim,
                dropout_rate=dropout_rate
            )
            fusion_in_dim = vis_dim + clinical_embed_dim
        elif self.mode == 'image_only':
            self.tabular_encoder = None
            fusion_in_dim = vis_dim
        else:
            raise ValueError("mode deve essere 'image_only' o 'multimodal'")
            
        # 3. Classifier Head finale (late fusion)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, img: torch.Tensor, clin: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Estrazione feature visive
        vis_features = self.visual_backbone(img)
        
        if self.mode == 'multimodal':
            if clin is None:
                raise ValueError("Vettore clinico richiesto in modalità multimodal.")
            clin_features = self.tabular_encoder(clin)
            # Concatenazione nello spazio latente (late fusion)
            fused_representation = torch.cat((vis_features, clin_features), dim=1)
            logits = self.classifier(fused_representation)
        else:
            logits = self.classifier(vis_features)
            
        return logits