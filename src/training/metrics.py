import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)

def compute_clinical_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calcola le metriche orientate allo screening dermatologico sbilanciato:
    - PR-AUC: metrica primaria per classi sbilanciate
    - Sensibilità (Recall classe 1): misura la minimizzazione dei falsi negativi
    - Balanced Accuracy
    - MCC (Matthews Correlation Coefficient)
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calcolo Precision-Recall AUC (PR-AUC)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    # Sensibilità clinica (Recall sulla classe maligna 1)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "pr_auc": float(pr_auc),
        "sensitivity": float(sensitivity),
        "balanced_acc": float(balanced_acc),
        "mcc": float(mcc),
        "confusion_matrix": cm
    }