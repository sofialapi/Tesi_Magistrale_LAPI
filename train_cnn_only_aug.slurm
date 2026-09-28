#!/bin/bash

# ============================================================
# Job GPU SLURM - Caso 1 (Augmented): ResNet-50 Image-Only
# ============================================================

#SBATCH --job-name=cnn_aug
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/cnn_only_aug_%j.out
#SBATCH --error=outputs/logs/cnn_only_aug_%j.err

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"
echo "========================================"

nvidia-smi

cd /mnt/sdb1/workspace/sofialapi/Tesi_Magistrale_LAPI
source /mnt/sdb1/workspace/sofialapi/venv_tesi/bin/activate

mkdir -p outputs/logs
mkdir -p outputs/checkpoints

# Esecuzione Caso 1 con Augmentation attiva
python run_tesla_training.py --case cnn_only --epochs 30 --batch_size 32 --lr 1e-4

echo "========================================"
echo "End:       $(date)"
echo "========================================"
