#!/bin/bash
#SBATCH --job-name=separate_datasets
#SBATCH --output=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/SLURM/create_final_dataset_lfm_2b_%j_output.txt
#SBATCH -o /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_outs/slurm-%j.out
#SBATCH --error=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH -c 4

module load anaconda/3
conda activate crs_env

dt=$(date '+%d/%m/%Y %H:%M:%S');


echo "$dt" + 'running experiment' + $1 + 'jobid' + $SLURM_JOB_ID >> /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_experiments.txt

python -u /home/mila/a/armin.moradi/CulturalDiscoverability/separate_datasets.py