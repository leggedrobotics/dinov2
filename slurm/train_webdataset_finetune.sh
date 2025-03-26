#!/bin/bash -l

#SBATCH --job-name=webd_clariden
#SBATCH --time=20:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=70
#SBATCH --account=a-a03
#SBATCH --output=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.err
#SBATCH --environment=vllm
#SBATCH --container-workdir=/users/patelm/ws/rsl/dinov2
# Load bashrc to ensure environment is properly sourced
source ~/.bashrc

# Export required paths
export PYTHONPATH=/users/patelm/ws/rsl/dinov2

# Move to the project directory
cd /users/patelm/ws/rsl/dinov2

# Install packages inside the container environment
#python -m pip install --user fvcore
#python -m pip install --user -r requirements.txt
#python -m pip install --user h5py
#python -m pip install --user webdataset
# Optional NCCL settings (if needed)
#export NCCL_TIMEOUT=1200

echo "Starting training"

echo $pwd
experiment=vitl14_finetune_depth_aug_32GPU_indoor_webd

# Run the training script inside the container
srun python -m dinov2.train.train \
  --exp-name ${experiment} \
  --config-file dinov2/configs/train/vitl14_finetune.yaml \
  --output-dir /capstor/store/cscs/swissai/a03/patelm/output_result/${experiment} \
  train.dataset_path=WebDatasetVision:root=/iopsstor/scratch/cscs/patelm/gfm_datasets/indoor_webd
