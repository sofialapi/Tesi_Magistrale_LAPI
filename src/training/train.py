# train.py
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

from src.config import PROCESSED_ISIC_DIR
from src.preprocessing.dataset_manager import (
    ClinicalMetadataProcessor,
    DermalMultimodalDataset,
    create_multimodal_dataloader
)
from src.models.multimodal_classifier import DermalClassifier
from src.training.losses import BinaryFocalLoss
from src.training.metrics import compute_clinical_metrics

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "outputs", "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def train_one_epoch(model, dataloader, optimizer, criterion, device, mode):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        if mode == 'multimodal':
            images, clinical, targets = batch
            images, clinical, targets = images.to(device), clinical.to(device), targets.to(device)
            logits = model(images, clinical)
        else:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device, mode):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if mode == 'multimodal':
                images, clinical, targets = batch
                images, clinical, targets = images.to(device), clinical.to(device), targets.to(device)
                logits = model(images, clinical)
            else:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                
            loss = criterion(logits, targets)
            running_loss += loss.item() * targets.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    metrics = compute_clinical_metrics(np.array(all_targets), np.array(all_probs))
    metrics["val_loss"] = val_loss
    return metrics

def run_stratified_kfold(
    df_metadata: pd.DataFrame,
    image_dir: str = PROCESSED_ISIC_DIR,
    case_study: str = "hybrid_multimodal",
    k_folds: int = 5,
    epochs: int = 30,
    warmup_epochs: int = 3,
    patience: int = 7,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Avvio 5-Fold CV | Caso Studio: [{case_study}] | Device: {device} ---")
    
    backbone_type = 'cnn' if 'cnn' in case_study else 'hybrid'
    mode = 'multimodal' if 'multimodal' in case_study else 'image_only'
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    y_all = df_metadata['target'].values
    image_ids_all = df_metadata['isic_id'].values
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_ids_all, y_all)):
        print(f"\n[Fold {fold + 1}/{k_folds}]")
        df_train, df_val = df_metadata.iloc[train_idx].copy(), df_metadata.iloc[val_idx].copy()
        
        # 1. Pipeline Isolamento Metadati
        if mode == 'multimodal':
            meta_processor = ClinicalMetadataProcessor()
            X_train_clin = meta_processor.fit_transform(df_train)
            X_val_clin = meta_processor.transform(df_val)
            clin_dim = X_train_clin.shape[1]
        else:
            X_train_clin, X_val_clin = None, None
            clin_dim = 0
        
        y_train = df_train['target'].values
        train_ids = df_train['isic_id'].values
        y_val = df_val['target'].values
        val_ids = df_val['isic_id'].values
        
        # 2. Bilanciamento delle classi reale con WeightedRandomSampler (no sintetizzazione artificiale)
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[y_train]
        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
            
        # 3. Dataset e DataLoader
        train_ds = DermalMultimodalDataset(
            image_ids=train_ids,
            labels=y_train,
            image_dir=image_dir,
            clinical_matrix=X_train_clin,
            mode=mode,
            is_training=True
        )
        val_ds = DermalMultimodalDataset(
            image_ids=val_ids,
            labels=y_val,
            image_dir=image_dir,
            clinical_matrix=X_val_clin,
            mode=mode,
            is_training=False
        )
        
        train_loader = create_multimodal_dataloader(
            train_ds, batch_size=batch_size, num_workers=4, sampler=train_sampler
        )
        val_loader = create_multimodal_dataloader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # 4. Modello, Loss (BinaryFocalLoss), Ottimizzatore AdamW
        model = DermalClassifier(
            backbone_type=backbone_type,
            mode=mode,
            clinical_dim=clin_dim,
            pretrained=True
        ).to(device)
        
        criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        
        # 5. Warmup + Cosine Annealing Scheduler
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        
        # 6. Early Stopping e Checkpoint
        best_val_loss = float('inf')
        best_metrics = None
        patience_counter = 0
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{case_study}_fold{fold+1}.pth")
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mode)
            val_metrics = evaluate(model, val_loader, criterion, device, mode)
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            print(f"  Epoca {epoch+1:02d}/{epochs:02d} [LR: {current_lr:.6f}] | TrLoss: {train_loss:.4f} | ValLoss: {val_metrics['val_loss']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | Recall: {val_metrics['sensitivity']:.4f}")
            
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_metrics = val_metrics
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [EARLY STOPPING] Nessun miglioramento della Val Loss per {patience} epoche consecutive al Fold {fold+1}.")
                    break
                    
        fold_results.append(best_metrics)
        
    print(f"\n=== Valutazione Media sui {k_folds} Fold ===")
    mean_prauc = np.mean([res["pr_auc"] for res in fold_results])
    mean_sens = np.mean([res["sensitivity"] for res in fold_results])
    mean_mcc = np.mean([res["mcc"] for res in fold_results])
    print(f"PR-AUC Medio: {mean_prauc:.4f} | Sensibilità Media: {mean_sens:.4f} | MCC Medio: {mean_mcc:.4f}")
    return fold_results
