# --- Experiment Constants --- #
SINGLE_SEED = 38210573
SEED_LIST = [SINGLE_SEED, 9491758, 2931009]
NUM_SAMPLES = 10  # How many hyperparameters samples will be taken into account
DATA_PATH = '/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/data'  # Path pointing at the data folder
GPU_PER_TRIAL = 1.0  # Ray Tune parameter,  how many gpus are allocated for a single trial experiment (https://docs.ray.io/en/releases-1.9.0/tune/api_docs/execution.html)
CPU_PER_TRIAL = 2  # Ray Tune parameter, how many cpus are allocated for a single trial experiment
# --- Training Constants --- #
MAX_PATIENCE = 5  # How many epochs without an improvement must pass before stopping the experiment
NUM_WORKERS = 2
# --- Evaluation Constants --- #
K_VALUES = [1, 3, 5, 10, 50]  # K value for the evaluation metrics
NEG_VAL = 99  # How many negative samples are considered during negative sampling
OPTIMIZING_METRIC = 'hit_ratio@10'  # Which metric will be used to assess during validation.
# --- Logger Constants --- #
WANDB_API_KEY = '35c9060c613a471ebd070e6e22727ddea1dbd739'  # Weight & Biases key for logging purposes
PROJECT_NAME = 'protomf'