#!/bin/bash -l

#SBATCH --job-name=webd_gen
#SBATCH --time=14:00:00
#SBATCH --nodes=1
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

python scripts/hdf5_to_webdataset.py


