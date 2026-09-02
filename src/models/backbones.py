import torch
import torch.nn as nn
import timm

class ResNetBackbone(nn.Module):
    """Backbone Convoluzionale Puro basato su ResNet-50 pre-addestrata."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Estrae le feature prima del classification head finale
        self.model = timm.create_model('resnet50', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.model.num_features  # 2048 per ResNet-50

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HybridVisionBackbone(nn.Module):
    """Backbone Ibrido CNN + ViT basato su MobileViT-S pre-addestrato."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # MobileViT unisce convoluzioni per dettagli locali e blocchi Transformer per il contesto globale
        self.model = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=0)
        self.feature_dim = self.model.num_features  # 640 per MobileViT-S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TabularMLP(nn.Module):
    """
    Encoder non lineare per feature tabulari cliniche.
    Proietta il vettore clinico in uno spazio di embedding denso.
    """
    def __init__(self, in_features: int = 9, embedding_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)