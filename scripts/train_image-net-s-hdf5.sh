#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --cpu-per-task 16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gpus=rtx_4090:4
#SBATCH --gres=gpumem:24G
#SBATCH --job-name=indoor-dino-depth
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err

source ~/.bashrc
conda activate dinov2
echo "Preparing the dataset"

export PYTHONPATH=/cluster/home/patelm/ws/rsl/dinov2

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.launch --nproc_per_node=4 dinov2/train/train.py --config-file dinov2/configs/train/vits16_short.yaml --output-dir /cluster/work/rsl/patelm/result/dino-depth-s-indoor-200ep train.dataset_path=GFMDataset:root=/cluster/scratch/patelm/indoor_datasets

