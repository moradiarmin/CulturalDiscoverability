# batch_size=313,
# data_path='/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/data/lfm2b-1mon',
# device='cpu',
# eval_neg_strategy='uniform',
# ft_ext_param={'embedding_dim': 93, 'ft_type': 'prototypes_double_tie'},

# 'user_ft_ext_param': {'cosine_type': 'shifted', 'ft_type': 'embedding_w', 'n_prototypes': 89, 'reg_batch_type': 'max', 'reg_proto_type': 'max', 'sim_batch_weight': 0.045405975375437474, 'sim_proto_weight': 0.003926141404662937, 'use_weight_matrix': False, 'out_dimension': 17}
# 'item_ft_ext_param': {'cosine_type': 'shifted', 'ft_type': 'embedding_w', 'n_prototypes': 17, 'reg_batch_type': 'max', 'reg_proto_type': 'max', 'sim_batch_weight': 0.0016067743528386975, 'sim_proto_weight': 1.2246719945476492, 'use_weight_matrix': False, 'out_dimension': 89}
# loss_func_aggr='mean',
# loss_func_name='sampled_softmax',
# n_epochs=30,
# neg_train=12,
# optim_param={'lr': 0.038545894730225266, 'optim': 'adagrad', 'wd': 0.0007576695565756338},
# rec_sys_param={'use_bias': 0}, seed=38210573, train_neg_strategy='uniform', val_batch_size=256

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
        'lr': 0.038545894730225266,
        'optim': 'adagrad',
        'wd': 0.0007576695565756338
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
        'ft_type': 'embedding_w',
        'n_prototypes': tune.randint(10, 100),
        'reg_batch_type': 'max',
        'reg_proto_type': 'max',
        'sim_batch_weight': 0.0016067743528386975,
        'sim_proto_weight': 1.2246719945476492,
        'use_weight_matrix': False,
        'out_dimension': 89},
        
    'user_ft_ext_param': {
        'cosine_type': 'shifted',
        'ft_type': 'embedding_w',
        'n_prototypes': tune.randint(10, 100),
        'reg_batch_type': 'max',
        'reg_proto_type': 'max',
        'sim_batch_weight': 0.045405975375437474,
        'sim_proto_weight': 0.003926141404662937,
        'use_weight_matrix': False, 'out_dimension': 17},
    },
}
