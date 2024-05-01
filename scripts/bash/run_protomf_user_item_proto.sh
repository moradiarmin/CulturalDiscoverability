#!/bin/bash
#SBATCH --job-name=run_protomf_user_item
#SBATCH --output=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/SLURM/run_protomf_user_item_%j_output.txt
#SBATCH --error=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/JOB_ERRORS/run_protomf_user_item_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=130:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -c 4

module load anaconda/3
conda activate protomf

dt=$(date '+%d/%m/%Y %H:%M:%S');

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "$dt running experiment $1 jobid $SLURM_JOB_ID" >> /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_experiments.txt

job_number=$SLURM_JOB_ID

script_path="/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/start.py"
python3 -u "$script_path" --model user_item_proto --dataset lfm2b-1mon
