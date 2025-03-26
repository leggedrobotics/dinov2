#!/bin/bash -l
 
#SBATCH --job-name=debug_clariden
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --account=a-a03
#SBATCH --output=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.err


source ~/.bashrc


export PYTHONPATH=/users/patelm/ws/rsl/dinov2
cd /users/patelm/ws/rsl/dinov2
pip install -r requriements.txt
pip install h5py

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=1200

echo "Starting training"

srun --environment=vllm python -m dinov2.train.train --exp-name cluster_trial --config-file dinov2/configs/train/vitl16_short.yaml --output-dir /capstor/store/cscs/swissai/a03/patelm/output_result train.dataset_path=GFMDataset:root=/iopsstor/store/cscs/swissai/a03/patelm/gfm_datasets


