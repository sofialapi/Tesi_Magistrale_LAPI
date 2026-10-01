import argparse
import os
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, recall_score
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class HAMDataset(Dataset):
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

def get_model(arch_type):
    if arch_type == 'cnn':
        return timm.create_model('resnet50', pretrained=True, num_classes=1)
    elif arch_type == 'hybrid':
        return timm.create_model('mobilevit_s', pretrained=True, num_classes=1)
    raise ValueError("Architettura sconosciuta")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['cnn', 'hybrid'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--csv_path', type=str, default='data/ham10000/ham10000_prepared.csv')
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Addestramento HAM10000: [{args.arch.upper()}] su {device} ===")

    df = pd.read_csv(args.csv_path)
    neg_count = sum(df['target'] == 0)
    pos_count = sum(df['target'] == 1)
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

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    global_best_auc = 0.0
    history_records = []

    for fold in range(1, 6):
        print(f"\n==================== Fold {fold}/5 ====================")
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]

        train_loader = DataLoader(HAMDataset(train_df, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(HAMDataset(val_df, val_tf), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = get_model(args.arch).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        best_pr_auc_fold = 0.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss = 0.0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * imgs.size(0)
            tr_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            preds, true_labels = [], []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * imgs.size(0)
                    preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                    true_labels.extend(targets.cpu().numpy().flatten())

            val_loss /= len(val_loader.dataset)
            precision, recall, _ = precision_recall_curve(true_labels, preds)
            pr_auc = auc(recall, precision)
            bin_preds = [1 if p >= 0.5 else 0 for p in preds]
            mcc = matthews_corrcoef(true_labels, bin_preds)
            rec = recall_score(true_labels, bin_preds, zero_division=0)

            history_records.append({
                'fold': fold, 'epoch': epoch,
                'train_loss': tr_loss, 'val_loss': val_loss,
                'pr_auc': pr_auc, 'recall': rec, 'mcc': mcc
            })

            if pr_auc > best_pr_auc_fold:
                best_pr_auc_fold = pr_auc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_{args.arch}_ham10000_fold{fold}.pth"))

            if pr_auc > global_best_auc:
                global_best_auc = pr_auc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_{args.arch}_ham10000.pth"))

            print(f"Epoca {epoch:02d}/{args.epochs} | TrLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {rec:.4f} | MCC: {mcc:.4f}")

    csv_out = os.path.join(args.output_dir, f"metrics_ham10000_{args.arch}.csv")
    pd.DataFrame(history_records).to_csv(csv_out, index=False)
    print(f"\n[FINE] Metriche registrate in: {csv_out}")

if __name__ == '__main__':
    main()
