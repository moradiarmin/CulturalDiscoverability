#!/bin/bash
#SBATCH --job-name=run_protomf_validation_evaluation
#SBATCH --output=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/SLURM/run_protomf_validation_evaluation_%j_output.txt
#SBATCH --error=/home/mila/a/armin.moradi/CulturalDiscoverability/logs/JOB_ERRORS/run_protomf_validation_evaluation_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH -c 2

module load anaconda/3
conda activate protomf

dt=$(date '+%d/%m/%Y %H:%M:%S');

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "$dt running experiment $1 jobid $SLURM_JOB_ID" >> /home/mila/a/armin.moradi/CulturalDiscoverability/logs/all_experiments.txt

job_number=$SLURM_JOB_ID

script_path="/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/evaluation_training_plots.py"
dataset_path="/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/data/lfm2b-1mon"
figs_path="/home/mila/a/armin.moradi/CulturalDiscoverability/logs/FIGS/${job_number}"
mkdir -p "$figs_path"


# experiment paths
k_1="/home/mila/a/armin.moradi/ray_results/user_item_proto_lfm2b-1mon_cn_38210573_2024-4-5_15-12-24.486415/user_item_proto_lfm2b-1mon_cn_38210573_36e2aa7f_6_batch_size=313,data_path=_home_mila_a_armin.moradi_CulturalDiscoverability_Proto_2024-04-06_00-36-19"
k_25="/home/mila/a/armin.moradi/ray_results/user_item_proto_lfm2b-1mon_cn_38210573_2024-3-28_13-1-22.594041/user_item_proto_lfm2b-1mon_cn_38210573_08bf1e42_3_batch_size=313,data_path=_home_mila_a_armin.moradi_CulturalDiscoverability_Proto_2024-03-28_17-06-24"
k_60="/home/mila/a/armin.moradi/ray_results/user_item_proto_lfm2b-1mon_cn_38210573_2024-3-28_13-1-22.594041/user_item_proto_lfm2b-1mon_cn_38210573_1b6f2c5f_7_batch_size=313,data_path=_home_mila_a_armin.moradi_CulturalDiscoverability_Proto_2024-03-28_22-07-34"

k_1_reg_item="/home/mila/a/armin.moradi/ray_results/user_item_proto_lfm2b-1mon_cn_38210573_2024-4-3_13-53-18.594856/user_item_proto_lfm2b-1mon_cn_38210573_12cc0949_1_batch_size=313,data_path=_home_mila_a_armin.moradi_CulturalDiscoverability_Proto_2024-04-03_13-53-27"


python3 -u "$script_path" --dataset_path "$dataset_path" --figs_path "$figs_path" --experiment_path "$k_1"
