#!/bin/bash -l

#SBATCH --job-name=dino_clariden
#SBATCH --time=12:00:00
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=70
#SBATCH --account=a144
#SBATCH --output=/iopsstor/scratch/cscs/patelm/output_slurm/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/patelm/output_slurm/%x_%j.err
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
experiment=vitl14_data_all_96GPU_3_channel_625k_webd

# Run the training script inside the container
srun python -m dinov2.train.train_depth_aug \
  --exp-name ${experiment} \
  --config-file dinov2/configs/train/vitl14.yaml \
  --output-dir /iopsstor/scratch/cscs/patelm/output_result/${experiment} \
  train.dataset_path=WebDatasetVisionPNG:root=/iopsstor/scratch/cscs/patelm/datasets_gfm