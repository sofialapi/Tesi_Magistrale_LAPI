#!/bin/bash

# ============================================================
# Job GPU SLURM - Caso 3: Multimodale (Immagini + Dati Clinici)
# ============================================================

#SBATCH --job-name=multimodal
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/multimodal_%j.out
#SBATCH --error=outputs/logs/multimodal_%j.err

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

# Esecuzione Caso 3 (Multimodale con Late Fusion)
python run_tesla_training.py --case cnn_multimodal --epochs 30 --batch_size 32 --lr 1e-4

echo "========================================"
echo "End:       $(date)"
echo "========================================"
