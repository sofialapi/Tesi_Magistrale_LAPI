#!/bin/bash
#SBATCH --job-name=vgg_bench
#SBATCH --output=outputs/vgg/slurm_pipeline_%j.log
#SBATCH --error=outputs/vgg/slurm_pipeline_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

./run_vgg_night_pipeline.sh
