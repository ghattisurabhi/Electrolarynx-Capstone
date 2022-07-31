#!/bin/bash
#SBATCH -A uvavoice      # your allocation account
#SBATCH -p gpu          # partition
#SBATCH --gres=gpu:4   # number of GPUs
#SBATCH -N 1            # number of nodes
#SBATCH -c 20         # number of cores
#SBATCH -t 24:00:00     # time
#SBATCH --mem=156GB 
module purge
module load singularity deeplabcut

singularity run --nv $CONTAINERDIR/deeplabcut-2.2.sif /scratch/sa9nc/Capstone/scripts/scripts.py
