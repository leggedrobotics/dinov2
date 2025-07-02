#!/bin/bash -l

#SBATCH --job-name=dino_clariden
#SBATCH --time=08:00:00
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=70
#SBATCH --account=a-a144
#SBATCH --output=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.err
#SBATCH --environment=vllm
#SBATCH --container-workdir=/users/patelm/ws/rsl/dinov2
# Load bashrc to ensure environment is properly sourced
source ~/.bashrc
ulimit -c 0

# Export required paths
export PYTHONPATH=/users/patelm/ws/rsl/dinov2

# Move to the project directory
cd /users/patelm/ws/rsl/dinov2

echo "Starting training"

echo $pwd
experiment=vitl14_depth_aug_96GPU_data_a_webd

# Run the training script inside the container
srun python -m dinov2.train.train \
  --exp-name ${experiment} \
  --config-file dinov2/configs/train/vitl14.yaml \
  --output-dir /capstor/store/cscs/swissai/a03/patelm/output_result/${experiment} \
  train.dataset_path=WebDatasetVisionPNG:root=/iopsstor/scratch/cscs/patelm/datasets_gfm/
