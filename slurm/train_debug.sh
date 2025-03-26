#!/bin/bash -l

#SBATCH --job-name=debug_clariden
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
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

# Optional NCCL settings (if needed)
#export NCCL_TIMEOUT=1200
export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)

echo "Starting training"

echo $pwd
experiment=cluster_trial_1N_1GPU_4w_py

#TORCHRUN_ARGS=\"
#   --node-rank=\${SLURM_PROCID} \
#   --master-addr=\${MASTER_ADDR} \
#   --master-port=\${MASTER_PORT} \
#   --nnodes=\${SLURM_NNODES} \
#   --nproc-per-node=\${SLURM_NTASKS_PER_NODE} \
#  \"


# Run the training script inside the container
torchrun --node-rank=${SLURM_PROCID} \
   --master-addr=${MASTER_ADDR} \
   --master-port=${MASTER_PORT} \
   --nnodes=${SLURM_NNODES} \
   --nproc-per-node=${SLURM_NTASKS_PER_NODE} \ 
   dinov2/train/train.py
  --exp-name ${experiment} \
  --config-file dinov2/configs/train/vitl16_short.yaml \
  --output-dir /capstor/store/cscs/swissai/a03/patelm/output_result/${experiment} \
  train.dataset_path=GFMDataset:root=/iopsstor/store/cscs/swissai/a03/patelm/gfm_datasets
