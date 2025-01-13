#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:24G
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err
#SBATCH --mem-per-cpu=4000
#SBATCH --time=4:00:00

source ~/.bashrc
conda activate dinov2

export PYTHONPATH=/cluster/home/patelm/ws/rsl/dinov2
cd /cluster/home/patelm/ws/rsl/dinov2

echo $CUDA_VISIBLE_DEVICES

echo "Starting training"

srun --gres=gpumem:24G python -m dinov2.train.train --exp-name dino-3090-2N-4GPU --config-file dinov2/configs/train/vits16_short.yaml --output-dir /cluster/work/rsl/patelm/result/debug_dino_s_2N_4GPU train.dataset_path=GFMDataset:root=/cluster/scratch/patelm/indoor_datasets

