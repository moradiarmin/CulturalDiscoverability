#!/bin/bash
#SBATCH --job-name=augment_track_samples
#SBATCH --output=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/SLURM/augment_track_samples_%j_output.txt
#SBATCH --error=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/JOB_ERRORS/job_errors_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 2

module load anaconda/3
conda activate crs_env

dt=$(date '+%d/%m/%Y %H:%M:%S');

echo "$dt running experiment $1 jobid $SLURM_JOB_ID" >> /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_experiments.txt

job_number=$SLURM_JOB_ID

script_path="/home/mila/a/armin.moradi/CulturalDiscoverability/scripts/py/augment_track_samples.py"
python -u "$script_path"
