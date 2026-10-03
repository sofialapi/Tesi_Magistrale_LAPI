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

class DermalAugmentor:
    def __init__(self, is_training=True):
        self.is_training = is_training
        if self.is_training:
            self.aug = transforms.Compose([
                transforms.RandomRotation(degrees=(-30, 30)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(-5, 5)),
                transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))
            ])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _sigmoid_correction(self, t: torch.Tensor, cutoff=0.5, gain=10.0) -> torch.Tensor:
        c = 1.0 / (1.0 + torch.exp(gain * (cutoff - t)))
        return (c - c.min()) / (c.max() - c.min() + 1e-6)

    def __call__(self, img_pil):
        t = transforms.functional.to_tensor(img_pil)
        if self.is_training:
            t = self.aug(t)
            if torch.rand(1).item() < 0.3:
                t = self._sigmoid_correction(t)
        return self.norm(t)

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
        return timm.create_model('vgg16', pretrained=True, num_classes=1)
    elif arch_type == 'hybrid':
        return timm.create_model('mobilevit_s', pretrained=True, num_classes=1)
    raise ValueError(f"Architettura sconosciuta: {arch_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['cnn', 'hybrid'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--csv_path', type=str, default='data/ham10000/ham10000_prepared.csv')
    parser.add_argument('--save_dir', type=str, default='outputs/vgg/checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs/vgg/metrics')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== HAM10000 DullRazor VGG Study: [{args.arch.upper()}] (Batch: {args.batch_size}, AMP: True) ===")

    df = pd.read_csv(args.csv_path)
    neg_count = sum(df['target'] == 0)
    pos_count = sum(df['target'] == 1)
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)

    train_tf = DermalAugmentor(is_training=True)
    val_tf = DermalAugmentor(is_training=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')
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
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
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
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_vgg_{args.arch}_ham10000_fold{fold}.pth"))

            if pr_auc > global_best_auc:
                global_best_auc = pr_auc
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_vgg_{args.arch}_ham10000.pth"))

            print(f"Epoca {epoch:02d}/{args.epochs} | TrLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {rec:.4f} | MCC: {mcc:.4f}")

        del model, optimizer
        torch.cuda.empty_cache()

    csv_out = os.path.join(args.output_dir, f"metrics_vgg_ham10000_{args.arch}.csv")
    pd.DataFrame(history_records).to_csv(csv_out, index=False)
    print(f"\n[FINE] Metriche registrate in: {csv_out}")

if __name__ == '__main__':
    main()
