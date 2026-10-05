import argparse
import os
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, recall_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class BCN20000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(row['target'], dtype=torch.float32)
        return image, target

def get_model(arch_type):
    if arch_type == 'cnn':
        # VGG-16 classico con testa singola logit binaria
        return timm.create_model('vgg16', pretrained=True, num_classes=1)
    elif arch_type == 'hybrid':
        # MobileViT-S
        return timm.create_model('mobilevit_s', pretrained=True, num_classes=1)
    raise ValueError(f"Architettura sconosciuta: {arch_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['cnn', 'hybrid'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='data/bcn20000')
    parser.add_argument('--save_dir', type=str, default='outputs/vgg/checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs/vgg/metrics')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== BCN20000 VGG Study: [{args.arch.upper()}] (Batch: {args.batch_size}, AMP: True) ===")

    csv_path = os.path.join(args.data_dir, 'train.csv')
    img_dir = os.path.join(args.data_dir, 'train')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    df = pd.read_csv(csv_path)
    col_name = 'image_name' if 'image_name' in df.columns else 'image'

    if 'target' in df.columns:
        df['target'] = df['target'].astype(int)
    elif 'MEL' in df.columns:
        df['target'] = df['MEL'].astype(int)
    else:
        raise KeyError("Colonna target o MEL non trovata nel CSV!")

    print("Mappatura file immagini su disco...")
    existing_files = {f: os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')}

    mapped_paths = []
    for raw_name in df[col_name]:
        base = str(raw_name).replace('.jpg', '')
        candidates = [
            f"{base}.jpg",
            f"{base}_downsampled.jpg",
            base.replace('_downsampled', '') + '.jpg'
        ]
        found = None
        for c in candidates:
            if c in existing_files:
                found = existing_files[c]
                break
        mapped_paths.append(found)

    df['filepath'] = mapped_paths
    initial_len = len(df)
    df = df.dropna(subset=['filepath']).reset_index(drop=True)
    neg_count = sum(df['target'] == 0)
    pos_count = sum(df['target'] == 1)
    print(f"Dataset filtrato: {len(df)} su {initial_len} campioni validi.")
    print(f"Distribuzione classi: Negativi={neg_count}, Melanomi={pos_count}")

    pos_weight = torch.tensor([neg_count / pos_count]).to(device)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')

    global_best_auc = 0.0
    history_records = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"\n==================== Fold {fold + 1}/5 ====================")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_loader = DataLoader(BCN20000Dataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(BCN20000Dataset(val_df, val_tf), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = get_model(args.arch).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

        best_pr_auc_fold = 0.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                tr_loss += loss.item() * imgs.size(0)
            tr_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            preds, true_labels = [], []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                    with torch.amp.autocast('cuda'):
                        outputs = model(imgs)
                        loss = criterion(outputs, targets)
                    val_loss += loss.item() * imgs.size(0)
                    batch_preds = torch.sigmoid(outputs)
                    batch_preds = torch.nan_to_num(batch_preds, nan=0.0)
                    preds.extend(batch_preds.cpu().numpy().flatten())
                    true_labels.extend(targets.cpu().numpy().flatten())

            val_loss /= len(val_loader.dataset)
            precision, recall, _ = precision_recall_curve(true_labels, preds)
            pr_auc = auc(recall, precision)
            bin_preds = [1 if p >= 0.5 else 0 for p in preds]
            mcc = matthews_corrcoef(true_labels, bin_preds)
            rec = recall_score(true_labels, bin_preds, zero_division=0)

            history_records.append({
                'fold': fold + 1,
                'epoch': epoch,
                'train_loss': tr_loss,
                'val_loss': val_loss,
                'pr_auc': pr_auc,
                'recall': rec,
                'mcc': mcc
            })

            if pr_auc > best_pr_auc_fold:
                best_pr_auc_fold = pr_auc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_vgg_{args.arch}_bcn20000_fold{fold+1}.pth"))

            if pr_auc > global_best_auc:
                global_best_auc = pr_auc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_vgg_{args.arch}_bcn20000.pth"))

            print(f"Epoca {epoch:02d}/{args.epochs} | TrLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {rec:.4f} | MCC: {mcc:.4f}")

        del model, optimizer
        torch.cuda.empty_cache()

    metrics_csv = os.path.join(args.output_dir, f"metrics_vgg_bcn20000_{args.arch}.csv")
    pd.DataFrame(history_records).to_csv(metrics_csv, index=False)
    print(f"\n[COMPLETATO] Metriche salvate in: {metrics_csv}")

if __name__ == '__main__':
    main()
