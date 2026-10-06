#!/bin/bash
#SBATCH --job-name=shap_isic
#SBATCH --output=outputs/shap/slurm_shap_%j.log
#SBATCH --error=outputs/shap/slurm_shap_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

cd "$SLURM_SUBMIT_DIR"
echo "Python: $(which python) | GPU: $CUDA_VISIBLE_DEVICES | inizio: $(date)"

python -m XAI_SHAP.shap_run && python -m XAI_SHAP.shap_plots

echo "fine: $(date)"
