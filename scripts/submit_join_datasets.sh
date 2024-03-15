#!/bin/bash
#SBATCH --job-name=join_datasets
#SBATCH --output=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/SLURM/artist_join_%j_output.txt
#SBATCH --error=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH -c 2

module load anaconda/3
conda activate crs_env

dt=$(date '+%d/%m/%Y %H:%M:%S');


echo "$dt" + 'running experiment' + $1 + 'jobid' + $SLURM_JOB_ID >> /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_experiments.txt

python -u /home/mila/a/armin.moradi/CulturalDiscoverability/join_datasets.py
