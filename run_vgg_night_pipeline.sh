#!/usr/bin/env bash
set -e

source /mnt/sdb1/workspace/sofialapi/venv_tesi/bin/activate
cd /mnt/sdb1/workspace/sofialapi/Tesi_Magistrale_LAPI

mkdir -p outputs/vgg/checkpoints outputs/vgg/metrics figure/vgg

echo "=========================================================="
echo "Avvio Pipeline Notturna VGG-16 vs MobileViT-S: $(date)"
echo "=========================================================="

# 1. VGG-16 su HAM10000
echo -e "\n[1/4] Addestramento VGG-16 (CNN only) su HAM10000..."
python train_vgg_ham.py --arch cnn > outputs/vgg/train_vgg_ham_cnn.log 2>&1
echo "[1/4] Finito: $(date)"

# 2. MobileViT-S su HAM10000
echo -e "\n[2/4] Addestramento MobileViT-S (Hybrid) su HAM10000..."
python train_vgg_ham.py --arch hybrid > outputs/vgg/train_vgg_ham_hybrid.log 2>&1
echo "[2/4] Finito: $(date)"

# 3. VGG-16 su BCN20000
echo -e "\n[3/4] Addestramento VGG-16 (CNN only) su BCN20000..."
python train_vgg_bcn20000.py --arch cnn > outputs/vgg/train_vgg_bcn_cnn.log 2>&1
echo "[3/4] Finito: $(date)"

# 4. MobileViT-S su BCN20000
echo -e "\n[4/4] Addestramento MobileViT-S (Hybrid) su BCN20000..."
python train_vgg_bcn20000.py --arch hybrid > outputs/vgg/train_vgg_bcn_hybrid.log 2>&1
echo "[4/4] Finito: $(date)"

echo -e "\n=========================================================="
echo "Tutti i 4 modelli addestrati con successo: $(date)"
echo "=========================================================="
