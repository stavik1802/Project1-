#!/bin/bash
#SBATCH --job-name=mpac_train
#SBATCH --output=logs/mpac_train_%j.out
#SBATCH --error=logs/mpac_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00  # Run up to 12 hours overnight
#SBATCH --partition=all  # Replace with your cluster's partition name
#SBATCH --gres=gpu:1        # If you need a GPU, remove if CPU-only
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stavos114@gmail.com

# Optional: Load required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate MPAC_310

# Ensure logs directory exists
mkdir -p logs


# Run your training script
#python -u -m MPAC.train_wrap --env_type energy_net --env_name energy_net --task_name energy_net --total_timesteps 500000 --runs 3 --energy_constraints battery --gamma 1.00 --env_horizon 48 --safety_budget 0.4 --actor_std_mult 1.0 --safe --safety_coeff 1.0 --robust --robust_type ramu --rob magnitude 0.001
python -m MPAC.eval --import_file mpac3 --import_path logs/models --energy_constraints_all
#1 mpac -battery ramu 0.001 budget 0.4  
# 2 crpo - thermal 
# 3 crpo - all 