#!/bin/bash

# --- Configuration Slurm ---
#SBATCH --job-name=QEC_Sim_All
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --qos=cpu

module load apptainer

mkdir -p logs

CONTAINER_NAME="squidasm-debian-12.sif"

echo "Start simulation: $(date)"
 
apptainer exec $CONTAINER_NAME python3.10 main.py --error all --prob 0.001

echo "End simulation: $(date)"