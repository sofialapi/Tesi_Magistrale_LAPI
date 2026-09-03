import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss binaria per mitigare il severo sbilanciamento delle classi.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Se i logits hanno shape [B, 2] (classificazione binaria con 2 neuroni di uscita)
        if logits.shape[1] == 2:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            p = torch.exp(-ce_loss)
            focal_loss = ((1.0 - p) ** self.gamma) * ce_loss
            
            # Ponderazione alfa per la classe positiva (1 = melanoma)
            alpha_weight = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            loss = alpha_weight * focal_loss
        else:
            # Caso sigmoide scalare singola [B, 1]
            probs = torch.sigmoid(logits.squeeze(1))
            targets = targets.float()
            bce = F.binary_cross_entropy_with_logits(logits.squeeze(1), targets, reduction='none')
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_factor * ((1.0 - p_t) ** self.gamma) * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss