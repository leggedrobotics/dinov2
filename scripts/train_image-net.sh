#!/bin/bash

#SBATCH -n 12
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=500G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:38G
#SBATCH --job-name=imagenet-dino
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err

source ~/.bashrc
conda activate dinov2
echo "Preparing the dataset"

export PYTHONPATH=/cluster/home/patelm/ws/rsl/dinov2

cd /cluster/home/patelm/ws/rsl/dinov2
python scripts/organize_data.py

echo "Preparing the metadata"

python scripts/imagenet_dataset_metadata.py

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.launch --nproc_per_node=4 dinov2/train/train.py --config-file dinov2/configs/train/vitl16_short.yaml --output-dir /cluster/work/rsl/patelm/result/ train.dataset_path=ImageNet:split=TRAIN:root=${TMPDIR}/imagenet-1k:extra=${TMPDIR}/imagenet-1k

python -m torch.distributed.launch --nproc_per_node=1 dinov2/eval/knn.py \
    --config-file /cluster/work/rsl/patelm/result/config.yaml \
    --pretrained-weights /cluster/work/rsl/patelm/result/eval/training_99999/teacher_checkpoint.pth \
    --output-dir /cluster/work/rsl/patelm/result/eval/training_99999/knn_10 \
    --train-dataset ImageNet:split=TRAIN:root=${TMPDIR}/imagenet-1k:extra=${TMPDIR}/imagenet-1k \
    --val-dataset ImageNet:split=VAL:root=${TMPDIR}/imagenet-1k:extra=${TMPDIR}/imagenet-1k
