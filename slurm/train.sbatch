#!/bin/bash
#SBATCH --job-name=rheidae-train
#SBATCH --account=isaac-utk0307
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=condo-slagergr
#SBATCH --qos=condo
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/home/jmcguig1/Rheidae
#SBATCH -o slurm/output/train-%j.out
#SBATCH -e slurm/output/train-%j.err

module load gcc/10.2.0
module load openmpi/4.0.3-gcc
module load hdf5/1.10.7-gcc
module load Python/3.9.10-gcc
module load cuda/11.4.2-gcc

# Activate your cluster venv (adjust path as needed)
source /nfs/home/jmcguig1/venv_Emu/bin/activate

# Ensure logs dir exists
mkdir -p .slurm/output

# Override Makefile VENV so install/train use the cluster venv
export VENV=/nfs/home/jmcguig1/venv_Emu
export PYTHONPATH=src

# Install deps (no-op if already installed) and run training
# srun make install
srun make train
