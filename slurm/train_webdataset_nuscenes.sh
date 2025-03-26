#!/bin/bash -l

#SBATCH --job-name=dinov2_nuscenes
#SBATCH --time=01:00:00
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
experiment=vitl16_depth_32GPU_nuscenes

# Run the training script inside the container
srun python -m dinov2.train.train \
  --exp-name ${experiment} \
  --config-file dinov2/configs/train/vits16_nuscenes.yaml \
  --output-dir /capstor/store/cscs/swissai/a03/patelm/output_result/${experiment} \
  train.dataset_path=WebDatasetVisionRange:root=/iopsstor/scratch/cscs/patelm/datasets_gfm/nuscenes_range_webd
