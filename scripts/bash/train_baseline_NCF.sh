#!/bin/bash
#SBATCH --job-name=train_baseline_NCF
#SBATCH --output=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/SLURM/train_baseline_%j_output.txt
#SBATCH --error=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/JOB_ERRORS/job_errors_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=2:10:00
#SBATCH --mem=256G
#SBATCH -c 1
#SBATCH --gres=gpu:1

module load anaconda/3
conda activate crs_env

dt=$(date '+%d/%m/%Y %H:%M:%S');

echo "$dt running experiment $1 jobid $SLURM_JOB_ID" >> /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_experiments.txt

job_number=$SLURM_JOB_ID
output_path="/home/mila/a/armin.moradi/CulturalDiscoverability/results/model_outputs_${job_number}/"
dataset_path="/home/mila/a/armin.moradi/scratch/data/LFM_2b_seperated_final/"
script_path="/home/mila/a/armin.moradi/CulturalDiscoverability/scripts/py/train_baseline_NCF.py"

mkdir -p "$output_path"
python -u "$script_path" --logs_path "$output_path" --dataset_path "$dataset_path"
