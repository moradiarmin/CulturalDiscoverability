import torch
from ray import tune

base_param = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 30,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
    'rec_sys_param': {'use_bias': 0},
}

base_hyper_params = {
    **base_param,
    'neg_train': 12,
    'train_neg_strategy': 'uniform',
    'loss_func_name': 'sampled_softmax',
    'batch_size': 313,
    'optim_param': {
        'lr': 0.04,
        'optim': 'adagrad',
        'wd': 0.0008
    },
}

proto_double_tie_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        'embedding_dim': 93,

    'item_ft_ext_param': {
        'cosine_type': 'shifted',
        'ft_type': 'prototypes',
        'n_prototypes': 89,
        'reg_batch_type': 'max',
        'reg_proto_type': 'max',
        'sim_batch_weight': 0.002,
        'sim_proto_weight': 1.22,
        'use_weight_matrix': False,
        'out_dimension': 89,
        'sim_proto_vectors_weight': tune.choice([0.0, 0.003, 0.01, 0.1, 1.0, 10.0]),
        'reg_proto_vectors_type': 'ortho', # regu
        'k': tune.choice([3, 12, 25, 40, 60, -1]), # -1 = inclusive, # post-process
        'initialize': tune.choice(['random', 'zero']) # init ['random', 'zero', 'on_init_points']
        },
        
    'user_ft_ext_param': {
        'cosine_type': 'shifted',
        'ft_type': 'prototypes',
        'n_prototypes': 89,
        'reg_batch_type': 'max',
        'reg_proto_type': 'max',
        'sim_batch_weight': 0.04,
        'sim_proto_weight': 0.004,
        'use_weight_matrix': False,
        'out_dimension': 89,
        'sim_proto_vectors_weight': tune.choice([0.0, 0.003, 0.01, 0.1, 1.0, 10.0]),
        'reg_proto_vectors_type': 'ortho',
        'k': tune.choice([3, 12, 25, 40, 60, -1]), # -1 = inclusive, # post-process
        'initialize': tune.choice(['random', 'zero'])
        }
    },
}

# initialize (initialize)
# regularizer coefficients (sim proto vec, reg proto vec)
# choose k (k)