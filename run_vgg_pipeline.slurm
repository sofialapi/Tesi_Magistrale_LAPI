#!/bin/bash
#SBATCH --job-name=vgg_bcn
#SBATCH --output=outputs/vgg/slurm_pipeline_%j.log
#SBATCH --error=outputs/vgg/slurm_pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

./run_vgg_bcn.sh
