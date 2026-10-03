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

class BCNRawDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(row['target'], dtype=torch.float32)
        return image, target

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING: BCN20000 | Image-Only Raw | ResNet-50 (CNN) su {device} ===")

    data_dir = 'data/bcn20000'
    csv_path = os.path.join(data_dir, 'train.csv')
    img_dir = os.path.join(data_dir, 'train')

    df = pd.read_csv(csv_path)
    df['target'] = df['target'].astype(int)

    # Associazione diretta al file RAW senza filtri
    existing_files = {f: os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')}
    mapped = []
    for raw_name in df['image_name']:
        base = str(raw_name).replace('.jpg', '')
        candidates = [f"{base}.jpg", f"{base}_downsampled.jpg"]
        found = next((existing_files[c] for c in candidates if c in existing_files), None)
        mapped.append(found)

    df['filepath'] = mapped
    df = df.dropna(subset=['filepath']).reset_index(drop=True)

    neg_count = sum(df['target'] == 0)
    pos_count = sum(df['target'] == 1)
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
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
    history = []
    global_best_auc = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"\n==================== Fold {fold + 1}/5 ====================")
        train_loader = DataLoader(BCNRawDataset(df.iloc[train_idx], train_tf), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(BCNRawDataset(df.iloc[val_idx], val_tf), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        model = timm.create_model('resnet50', pretrained=True, num_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        best_fold_auc = 0.0

        for epoch in range(1, 21):
            model.train()
            tr_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * imgs.size(0)
            tr_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            preds, trues = [], []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                    out = model(imgs)
                    val_loss += criterion(out, targets).item() * imgs.size(0)
                    preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
                    trues.extend(targets.cpu().numpy().flatten())

            val_loss /= len(val_loader.dataset)
            precision, recall, _ = precision_recall_curve(trues, preds)
            pr_auc = auc(recall, precision)
            bin_preds = [1 if p >= 0.5 else 0 for p in preds]
            rec = recall_score(trues, bin_preds, zero_division=0)
            mcc = matthews_corrcoef(trues, bin_preds)

            history.append({
                'fold': fold + 1, 'epoch': epoch,
                'train_loss': tr_loss, 'val_loss': val_loss,
                'pr_auc': pr_auc, 'recall': rec, 'mcc': mcc
            })

            if pr_auc > best_fold_auc:
                best_fold_auc = pr_auc
                torch.save(model.state_dict(), f"outputs/checkpoints/raw/best_BCN_raw_cnn_fold{fold+1}.pth")
            if pr_auc > global_best_auc:
                global_best_auc = pr_auc
                torch.save(model.state_dict(), "outputs/checkpoints/raw/best_BCN_raw_cnn.pth")

            print(f"Epoca {epoch:02d}/20 | TrLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {rec:.4f} | MCC: {mcc:.4f}")

    pd.DataFrame(history).to_csv("outputs/raw/metrics_BCN_raw_cnn.csv", index=False)
    print("\n[COMPLETATO] Salvato in outputs/raw/metrics_BCN_raw_cnn.csv")

if __name__ == '__main__':
    main()
