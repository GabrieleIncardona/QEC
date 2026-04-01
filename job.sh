#!/bin/bash

# --- Configuration Slurm ---
#SBATCH --job-name=QEC_Sim_All        # Name of the job
#SBATCH --output=logs/sim_%j.out      # Output file (%j inserts the job ID)
#SBATCH --error=logs/sim_%j.err       # Error file
#SBATCH --nodes=1                     # Usually SquidASM runs on a single node (multi-core)
#SBATCH --ntasks=1                    # A single main task
#SBATCH --cpus-per-task=4            # Number of cores (useful for the simulation backend)
#SBATCH --mem=16G                     # Memoria RAM (regola in base a block_size B)
#SBATCH --time=02:00:00               # Limite tempo (HH:MM:SS)
#SBATCH --partition=cpu          # Name of the partition (changes based on your HPC)
#SBATCH --qos=cpu

# --- Caricamento Ambiente ---
# Carica il modulo python se necessario (dipende dal cluster)
# module load python/3.10 
module load gcc/9.5.0
# Attiva il tuo virtual environment dove hai installato NetQASM e SquidASM
# Assicurati di averlo creato precedentemente con: python -m venv venv
source venv/bin/activate

# Crea la cartella log se non esiste
mkdir -p logs

# --- Esecuzione ---
echo "Start simulation: $(date)"

# Eseguo lo script principale. 
# Passiamo l'argomento 'all' se il tuo main.py lo accetta tramite argparse
python main.py --error all --prob 0.01

echo "Simulazione completata: $(date)"