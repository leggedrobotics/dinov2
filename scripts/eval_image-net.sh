#!/bin/bash

#SBATCH -n 12
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=200G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20G
#SBATCH --job-name=imagenet-dino-eval
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

python -m torch.distributed.launch --nproc_per_node=1 dinov2/eval/knn.py \
    --config-file /cluster/work/rsl/patelm/result/dino-depth/config.yaml \
    --pretrained-weights /cluster/work/rsl/patelm/result/dino-depth/eval/training_112499/teacher_checkpoint.pth \
    --output-dir /cluster/work/rsl/patelm/result/dino-depth/eval/training_112499/knn_depth \
    --train-dataset ImageNet:split=TRAIN:root=${TMPDIR}/imagenet-1k:extra=${TMPDIR}/imagenet-1k \
    --val-dataset ImageNet:split=VAL:root=${TMPDIR}/imagenet-1k:extra=${TMPDIR}/imagenet-1k
