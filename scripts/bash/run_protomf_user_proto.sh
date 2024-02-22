#!/bin/bash
#SBATCH --job-name=run_protomf_user
#SBATCH --output=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/SLURM/run_protomf_user_%j_output.txt
#SBATCH --error=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/JOB_ERRORS/run_protomf_user_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=14:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -c 2

module load anaconda/3
conda activate protomf

dt=$(date '+%d/%m/%Y %H:%M:%S');

echo "$dt running experiment $1 jobid $SLURM_JOB_ID" >> /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_experiments.txt

job_number=$SLURM_JOB_ID

script_path="/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/start.py"
python3 -u "$script_path" --model user_proto --dataset lfm2b-1mon
