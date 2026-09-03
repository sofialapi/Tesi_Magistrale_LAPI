import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold

from src.config import PROCESSED_ISIC_DIR
from src.preprocessing.dataset_manager import (
    ClinicalMetadataProcessor,
    DermalMultimodalDataset,
    create_multimodal_dataloader
)
from src.preprocessing.balancer import TabularSMOTEBalancer
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
    case_study: str = "hybrid_multimodal",  # 'cnn_only', 'hybrid_only', 'cnn_multimodal', 'hybrid_multimodal'
    k_folds: int = 5,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Avvio 5-Fold CV | Caso Studio: [{case_study}] | Device: {device} ---")
    
    # Mappatura parametri modello
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
        meta_processor = ClinicalMetadataProcessor()
        X_train_clin = meta_processor.fit_transform(df_train)
        X_val_clin = meta_processor.transform(df_val)
        clin_dim = X_train_clin.shape[1]
        
        y_train = df_train['target'].values
        train_ids = df_train['isic_id'].values
        y_val = df_val['target'].values
        val_ids = df_val['isic_id'].values
        
        # 2. Applicazione SMOTE su Training Fold (se multimodale)
        if mode == 'multimodal':
            smote = TabularSMOTEBalancer()
            X_train_clin, y_train_res = smote.balance(X_train_clin, y_train)
            # Se SMOTE sovracampiona, duplichiamo coerentemente gli ID immagine per il training
            if len(y_train_res) > len(train_ids):
                extra_needed = len(y_train_res) - len(train_ids)
                minority_ids = train_ids[y_train == 1]
                sampled_extras = np.random.choice(minority_ids, size=extra_needed, replace=True)
                train_ids = np.concatenate([train_ids, sampled_extras])
            y_train = y_train_res
            
        # 3. Dataset e DataLoader
        train_ds = DermalMultimodalDataset(
            image_ids=train_ids,
            labels=y_train,
            image_dir=image_dir,
            clinical_matrix=X_train_clin if mode == 'multimodal' else None,
            mode=mode,
            is_training=True
        )
        val_ds = DermalMultimodalDataset(
            image_ids=val_ids,
            labels=y_val,
            image_dir=image_dir,
            clinical_matrix=X_val_clin if mode == 'multimodal' else None,
            mode=mode,
            is_training=False
        )
        
        train_loader = create_multimodal_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = create_multimodal_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # 4. Istanziazione Modello, Loss, Ottimizzatore
        model = DermalClassifier(
            backbone_type=backbone_type,
            mode=mode,
            clinical_dim=clin_dim,
            pretrained=True
        ).to(device)
        
        criterion = BinaryFocalLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_prauc = 0.0
        best_metrics = None
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mode)
            val_metrics = evaluate(model, val_loader, criterion, device, mode)
            scheduler.step()
            
            if val_metrics["pr_auc"] >= best_prauc:
                best_prauc = val_metrics["pr_auc"]
                best_metrics = val_metrics
                # Salvataggio del checkpoint ottimale per il fold corrente
                ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{case_study}_fold{fold+1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                
            print(f"  Epoca {epoch+1:02d}/{epochs:02d} | TrLoss: {train_loss:.4f} | ValLoss: {val_metrics['val_loss']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | Recall: {val_metrics['sensitivity']:.4f}")
            
        fold_results.append(best_metrics)
        
    print(f"\n=== Valutazione Media sui {k_folds} Fold ===")
    mean_prauc = np.mean([res["pr_auc"] for res in fold_results])
    mean_sens = np.mean([res["sensitivity"] for res in fold_results])
    mean_mcc = np.mean([res["mcc"] for res in fold_results])
    print(f"PR-AUC Medio: {mean_prauc:.4f} | Sensibilità Media: {mean_sens:.4f} | MCC Medio: {mean_mcc:.4f}")
    return fold_results