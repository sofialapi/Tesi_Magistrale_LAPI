#!/bin/bash

# ==============================================================================
# Job GPU SLURM - Sequenza Notturna: Caso 2, Caso 3 e Caso 4
# ==============================================================================

#SBATCH --job-name=night_train
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/night_run_%j.out
#SBATCH --error=outputs/logs/night_run_%j.err

echo "========================================"
echo "Avvio Pipeline Sequenziale Notturna"
echo "Job ID:    $SLURM_JOB_ID"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Data:      $(date)"
echo "========================================"

cd /mnt/sdb1/workspace/sofialapi/Tesi_Magistrale_LAPI
source /mnt/sdb1/workspace/sofialapi/venv_tesi/bin/activate

mkdir -p outputs/logs
mkdir -p outputs/checkpoints

# Funzione per estrarre il CSV dal log
extract_metrics() {
    local log_file=$1
    local out_csv=$2
    python -c "
import re, os, pandas as pd
records = []
current_fold = 0
pattern = re.compile(r'Epoca\s+(\d+)/\d+.*?TrLoss:\s+([\d\.]+)\s+\|\s+ValLoss:\s+([\d\.]+)\s+\|\s+PR-AUC:\s+([\d\.]+)\s+\|\s+Recall:\s+([\d\.]+)')
if os.path.exists('$log_file'):
    with open('$log_file', 'r') as f:
        for line in f:
            if '[Fold ' in line:
                current_fold = int(line.split('[Fold ')[1].split('/')[0])
            m = pattern.search(line)
            if m:
                e, tr, val, pr, rec = m.groups()
                records.append({'fold': current_fold, 'epoch': int(e), 'train_loss': float(tr), 'val_loss': float(val), 'pr_auc': float(pr), 'recall': float(rec)})
    pd.DataFrame(records).to_csv('$out_csv', index=False)
    print(f'[METRICHE] Esportate con successo in $out_csv ({len(records)} record)')
"
}

# ------------------------------------------------------------------------------
# 1. CASO 2: Hybrid Only (MobileViT-S Image-Only)
# ------------------------------------------------------------------------------
echo -e "\n>>> AVVIO CASO 2: hybrid_only ($(date)) <<<"
python run_tesla_training.py --case hybrid_only --epochs 30 --batch_size 32 --lr 1e-4 2>&1 | tee outputs/logs/hybrid_only_${SLURM_JOB_ID}.out
extract_metrics "outputs/logs/hybrid_only_${SLURM_JOB_ID}.out" "outputs/metrics_hybrid_only.csv"

# ------------------------------------------------------------------------------
# 2. CASO 3: CNN Multimodal (ResNet-50 + Tabulare)
# ------------------------------------------------------------------------------
echo -e "\n>>> AVVIO CASO 3: cnn_multimodal ($(date)) <<<"
python run_tesla_training.py --case cnn_multimodal --epochs 30 --batch_size 32 --lr 1e-4 2>&1 | tee outputs/logs/multimodal_${SLURM_JOB_ID}.out
extract_metrics "outputs/logs/multimodal_${SLURM_JOB_ID}.out" "outputs/metrics_multimodal.csv"

# ------------------------------------------------------------------------------
# 3. CASO 4: Hybrid Multimodal (MobileViT-S + Tabulare)
# ------------------------------------------------------------------------------
echo -e "\n>>> AVVIO CASO 4: hybrid_multimodal ($(date)) <<<"
python run_tesla_training.py --case hybrid_multimodal --epochs 30 --batch_size 32 --lr 1e-4 2>&1 | tee outputs/logs/hybrid_multimodal_${SLURM_JOB_ID}.out
extract_metrics "outputs/logs/hybrid_multimodal_${SLURM_JOB_ID}.out" "outputs/metrics_hybrid_multimodal.csv"

echo "========================================"
echo "Completamento sequenza: $(date)"
echo "========================================"

