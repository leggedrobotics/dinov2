#!/bin/bash

#SBATCH -n 16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=200G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:38G
#SBATCH --job-name=imagenet-dino-depth
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err

source ~/.bashrc
conda activate dinov2
echo "Preparing the dataset"

export PYTHONPATH=/cluster/home/patelm/ws/rsl/dinov2

cd /cluster/home/patelm/ws/rsl/dinov2
python scripts/organize_data_depth.py

echo "Preparing the metadata"

python scripts/imagenet_dataset_metadata.py

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.launch --nproc_per_node=4 dinov2/train/train.py --config-file dinov2/configs/train/vits16_short.yaml --output-dir /cluster/work/rsl/patelm/result/dino-depth-s train.dataset_path=ImageNet:split=TRAIN:root=${TMPDIR}/imagenet-1k:extra=${TMPDIR}/imagenet-1k
