#!/bin/bash

# ============================================================
# Job GPU SLURM - BCN20000: CNN Only (ResNet-50)
# ============================================================

#SBATCH --job-name=bcn_cnn
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=outputs/logs/bcn_cnn_%j.out
#SBATCH --error=outputs/logs/bcn_cnn_%j.err

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

# Esecuzione CNN pura (ResNet-50) su BCN20000
python train_bcn20000.py --arch cnn --epochs 20 --batch_size 64 --lr 1e-4

echo "========================================"
echo "End:       $(date)"
echo "========================================"
