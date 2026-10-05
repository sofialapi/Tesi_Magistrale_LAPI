#!/usr/bin/env bash
set -e

source /mnt/sdb1/workspace/sofialapi/venv_tesi/bin/activate
cd /mnt/sdb1/workspace/sofialapi/Tesi_Magistrale_LAPI

echo "=========================================================="
echo "Avvio Pipeline BCN20000 (VGG-16 vs MobileViT-S): $(date)"
echo "=========================================================="

# 1. VGG-16 su BCN20000
echo -e "\n[1/2] Addestramento VGG-16 su BCN20000..."
python train_vgg_bcn20000.py --arch cnn --batch_size 32 > outputs/vgg/train_vgg_bcn_cnn.log 2>&1
echo "[1/2] Finito: $(date)"

# 2. MobileViT-S su BCN20000
echo -e "\n[2/2] Addestramento MobileViT-S su BCN20000..."
python train_vgg_bcn20000.py --arch hybrid --batch_size 32 > outputs/vgg/train_vgg_bcn_hybrid.log 2>&1
echo "[2/2] Finito: $(date)"

echo -e "\n=========================================================="
echo "Completati tutti i modelli BCN20000: $(date)"
echo "=========================================================="
