import pandas as pd
import os
import torch
from rec_sys.tester import Tester
from experiment_helper import load_data
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utilities.eval import Evaluator
import experiment_helper
from utilities.explanations_utils import tsne_plot
import os

# parser

parser = argparse.ArgumentParser(description="Argparser for modifying the training settings")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--figs_path", type=str, help="Where to save all the plots and figures and tables?")
parser.add_argument("--experiment_path", type=str, help="Path to experiment (tested model)")

args = parser.parse_args()

print('args', args)
print('dataset_path:', args.dataset_path)
print('experiment_path:', args.experiment_path)
print('figs_path:', args.figs_path)

figs_path = args.figs_path
dataset_path = args.dataset_path
experiment_path = args.experiment_path

users = pd.read_csv(os.path.join(dataset_path, 'users.tsv'), sep='\t')
users.rename(columns={'user_id': 'old_user_id'}, inplace=True)
test_dataset_path = dataset_path + '/listening_history_test.csv'
test_data = pd.read_csv(test_dataset_path)
test_data_user_augmented = test_data.merge(users, on='old_user_id')
tracks = pd.read_csv(os.path.join(dataset_path, 'tracks_augmented.csv'))
tracks.rename(columns={'track_id': 'old_item_id'}, inplace=True)
test_data_user_item_augmented = test_data_user_augmented.merge(tracks, on='old_item_id')
test_data_user_item_augmented.rename(columns={'country_x': 'user_country', 'age':'user_age', 'gender_x': 'user_gender', 'artist': 'artist_name', 'track':'track_name', 'gender_y':'artist_gender', 'country_y':'artist_country'}, inplace=True)
train_dataset = pd.read_csv(os.path.join(dataset_path, 'listening_history_train.csv'))
test_data_user_item_augmented['user_num_interactions'] = train_dataset.groupby('user_id').size()[test_data_user_item_augmented['user_id']].values
test_data_user_item_augmented['item_num_interactions'] = train_dataset.groupby('item_id').size()[test_data_user_item_augmented['item_id']].values

# load model

config_path = os.path.join(experiment_path, 'params.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    
with open(f'{figs_path}/config.json', 'w') as f:
    print(f'dump {f} to {figs_path}')
    json.dump(config, f)

config['device'] = 'cpu'
config_dict = config.copy()
config = argparse.Namespace(**config)
data_loaders_dict = load_data(config, is_train=False)
model_load_path = os.path.join(experiment_path, 'checkpoint_000000/best_model.pth')
tester = Tester(data_loaders_dict['test_loader'], config, model_load_path)
model = tester.model

user_prototype_model = model.user_feature_extractor.model_1
item_prototype_model = model.item_feature_extractor.model_1

user_embeddings = user_prototype_model.embedding_ext.embedding_layer.weight[:]
user_prototypes = user_prototype_model.prototypes

item_embeddings = item_prototype_model.embedding_ext.embedding_layer.weight[:]
item_prototypes = item_prototype_model.prototypes

print('CONFIG!', config)
item_model_k = config.ft_ext_param['item_ft_ext_param']['k']

try:
    user_model_k = config.ft_ext_param['k'] # there is a bug in the config file, the user model k is stored in the wrong place
except:
    user_model_k = config.ft_ext_param['user_ft_ext_param']['k']

# VISUALIZATION

# 1.1 DISTANCE TO PROTOTYPES

def distance_to_prototypes(indices, prototypes, embeddings, mode='average', k=5):

    def average_distance_to_prototypes(idx):
        cosine_similarity = F.cosine_similarity(embeddings[idx].unsqueeze(0), prototypes, dim=1)
        return 1 - cosine_similarity.mean().item()

    def distance_to_closest_prototype(idx):
        closest_prototype_index = F.cosine_similarity(embeddings[idx].unsqueeze(0), prototypes, dim=1).argmax().item()
        cosine_similarity = F.cosine_similarity(embeddings[idx].unsqueeze(0), prototypes[closest_prototype_index], dim=1)
        return 1 - cosine_similarity.mean().item() 
        
    def distance_to_top_k_closest_prototypes(idx):
        closest_prototype_indices = F.cosine_similarity(embeddings[idx].unsqueeze(0), prototypes, dim=1).topk(k).indices
        cosine_similarity = F.cosine_similarity(embeddings[idx].unsqueeze(0), prototypes[closest_prototype_indices], dim=1)
        return 1 - cosine_similarity.mean().item()
    
    if mode == 'average':
        distance_function = average_distance_to_prototypes
    elif mode == 'closest':
        distance_function = distance_to_closest_prototype
    elif mode == 'top_k':
        distance_function = distance_to_top_k_closest_prototypes

    average_distances = []
    for idx in indices:
        average_distances.append(distance_function(idx))
    return np.array(average_distances)

male_user_ids = test_data_user_item_augmented.loc[test_data_user_item_augmented['user_gender']=='m']['user_id']
female_user_ids = test_data_user_item_augmented.loc[test_data_user_item_augmented['user_gender']=='f']['user_id']
user_ks = np.arange(1, user_prototypes.shape[0], 5)

male_distances = []
female_distances = []
for k in user_ks:
    male_distances.append(distance_to_prototypes(male_user_ids, user_prototypes, user_embeddings, mode='top_k', k=k).mean())
    female_distances.append(distance_to_prototypes(female_user_ids, user_prototypes, user_embeddings, mode='top_k', k=k).mean())

plt.figure()
plt.plot(user_ks, male_distances, 'b', label=f'Male Distances to k nearest prototypes, trained with k = {user_model_k}')
plt.plot(user_ks, female_distances, 'r', label=f'Female Distances to k nearest prototypes, trained with k = {user_model_k}')
plt.legend()
plt.savefig(f'{figs_path}/user_gender_distances.pdf')
plt.show()

country_counts = test_data_user_item_augmented.groupby(['user_country']).size().sort_values(ascending=False)

pop_countries = list(country_counts[:3].index) 
mid_countries = list(country_counts[90:93].index)

pop_countries_ids = test_data_user_item_augmented[test_data_user_item_augmented['user_country'].isin(pop_countries)]['user_id']
mid_countries_ids = test_data_user_item_augmented[test_data_user_item_augmented['user_country'].isin(mid_countries)]['user_id']

pop_distances = []
mid_distances = []
for k in user_ks:
    pop_distances.append(distance_to_prototypes(pop_countries_ids, user_prototypes, user_embeddings, mode='top_k', k=k).mean())
    mid_distances.append(distance_to_prototypes(mid_countries_ids, user_prototypes, user_embeddings, mode='top_k', k=k).mean())
    
country_counts = test_data_user_item_augmented.groupby(['user_country']).size().sort_values(ascending=False)

plt.figure()
plt.plot(user_ks, pop_distances, 'b', label=f'Pop Distances to k nearest prototypes, trained with k = {user_model_k}')
plt.plot(user_ks, mid_distances, 'r', label=f'Mid Distances to k nearest prototypes, trained with k = {user_model_k}')
plt.legend()
plt.savefig(f'{figs_path}/user_country_distances.pdf')
plt.show()

test_data_user_item_augmented['user_num_interactions'].describe(percentiles=[0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.99])

user_ids_arr_popularity_groups = []

plt.figure()
percentiles = {25: 57, 50: 130, 75: 245, 90: 395, 99: 749, 100: 1347}
per = [0] + list(percentiles.values())
percentilesss = list(percentiles.keys())
for i in range(len(per) - 1):
    user_ids_arr_popularity_groups.append(test_data_user_item_augmented.loc[(test_data_user_item_augmented['user_num_interactions'] > per[i]) & (test_data_user_item_augmented['user_num_interactions'] < per[i + 1])]['user_id'])

distances_arr = [[] for _ in range(len(user_ids_arr_popularity_groups))]
for user_group in range(len(user_ids_arr_popularity_groups)):
    for k in user_ks:
        distances_arr[user_group].append(distance_to_prototypes(user_ids_arr_popularity_groups[user_group], user_prototypes, user_embeddings, mode='top_k', k=k).mean())
distances_arr = np.array(distances_arr)
for i in range(len(user_ids_arr_popularity_groups)):
    plt.plot(user_ks, distances_arr[i]**5, label=f'Popularity Group Percentile - {percentilesss[i]}: {percentiles[percentilesss[i]]} Interactions, trained with k = {user_model_k}')
plt.legend()
plt.savefig(f'{figs_path}/user_popularity_groups_distances.pdf')
plt.show()

test_data_user_item_augmented['item_num_interactions'].describe(percentiles=[0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.99])

item_ks = np.arange(1, item_prototypes.shape[0], 5)

male_item_ids = test_data_user_item_augmented.loc[test_data_user_item_augmented['artist_gender']=='Male']['item_id']
female_item_ids = test_data_user_item_augmented.loc[test_data_user_item_augmented['artist_gender']=='Female']['item_id']

male_distances = []
female_distances = []
for k in item_ks:
    male_distances.append(distance_to_prototypes(male_item_ids, item_prototypes, item_embeddings, mode='top_k', k=k).mean())
    female_distances.append(distance_to_prototypes(female_item_ids, item_prototypes, item_embeddings, mode='top_k', k=k).mean())
plt.figure()
plt.plot(item_ks, male_distances, 'b', label=f'Male Distances to k nearest prototypes, trained with k = {item_model_k}')
plt.plot(item_ks, female_distances, 'r', label=f'Female Distances to k nearest prototypes, trained with k = {item_model_k}')
plt.legend()
plt.savefig(f'{figs_path}/item_gender_distances.pdf')
plt.show()

item_ids_arr_popularity_groups = []

percentiles = {25: 12, 50: 21, 75: 45, 90: 89, 99: 265, 100:689}
percentilesss = list(percentiles.keys())
per = [0] + list(percentiles.values())

for i in range(len(per) - 1):
    item_ids_arr_popularity_groups.append(test_data_user_item_augmented.loc[(test_data_user_item_augmented['item_num_interactions'] > per[i]) & (test_data_user_item_augmented['item_num_interactions'] <= per[i + 1])]['item_id'])
    
distances_arr = [[] for _ in range(len(item_ids_arr_popularity_groups))]
for item_group in range(len(item_ids_arr_popularity_groups)):
    for k in item_ks:
        distances_arr[item_group].append(distance_to_prototypes(item_ids_arr_popularity_groups[item_group], item_prototypes, item_embeddings, mode='top_k', k=k).mean())
distances_arr = np.array(distances_arr)

plt.figure()
ks = np.arange(1, item_prototypes.shape[0], 5)
for i in range(len(item_ids_arr_popularity_groups)):
    plt.plot(ks, distances_arr[i], label=f'Popularity Group Percentile - {percentilesss[i]}: {percentiles[percentilesss[i]]} Interactions, trained with k = {item_model_k}')
plt.legend()
plt.savefig(f'{figs_path}/item_popularity_groups_distances.pdf')
plt.show()




def evaluate_for_item_subgroups(model, column='user_gender', subgroups=['m', 'f']):

    model.eval()
    test_loader = experiment_helper.load_data(config, is_train=False)['test_loader']
    groups = {subgroup:test_data_user_item_augmented[test_data_user_item_augmented[column] == subgroup]['user_id'] for subgroup in subgroups}
    group_losses = {}
    group_evals = {}

    for group_name, subgroup in groups.items():
        test_loss = 0
        eval = Evaluator(0)
        for u_idxs, i_idxs, labels in test_loader:
            mask = pd.Series(u_idxs).isin(subgroup)
            u_idxs, i_idxs, labels = u_idxs[mask], i_idxs[mask], labels[mask]
            
            if len(u_idxs) == 0:
                continue
            eval.n_users += len(u_idxs)
            out = model(u_idxs, i_idxs).detach()
            test_loss += model.loss_func(out, labels).item()

            out = nn.Sigmoid()(out)
            out = out.to('cpu')

            eval.eval_batch(out, group_name)

        group_losses[group_name] = test_loss
        group_evals[group_name] = eval

    return group_losses, group_evals


feature_column = 'user_country'
feature_subgroups = ['US', 'UK', 'IR', 'IT', 'CA', 'UA']
losses, metrics = evaluate_for_user_subgroups(model, column=feature_column, subgroups=feature_subgroups)
metrics_dic = {}
for subgroup in feature_subgroups:
    metrics_dic[subgroup] = metrics[subgroup].get_results()

countries = metrics_dic.keys()

keys = list(metrics_dic.keys())
metrics = list(metrics_dic[keys[0]].keys())

plt.figure(figsize=(12, 8))  # Adjust size here
bar_width = 0.07

num_metrics = len(metrics)
index = range(len(keys))

fig, ax = plt.subplots()
for i in range(num_metrics):
    metric_values = [metrics_dic[key][metrics[i]] for key in keys]
    ax.bar([x + i * bar_width for x in index], metric_values, bar_width, label=metrics[i])
ax.set_xlabel('Keys')
ax.set_ylabel('Metrics')
ax.set_title(f'Metrics for Different Keys, trained with k = {user_model_k}')
ax.set_xticks([x + 0.3 for x in index])
ax.set_xticklabels(keys)
ax.legend()

plt.tight_layout()
plt.savefig(f'{figs_path}/Metrics for Different countries, trained with k = {user_model_k}.pdf')
plt.show()

feature_column = 'user_gender'
feature_subgroups = ['m', 'f']
losses, metrics = evaluate_for_user_subgroups(model)
metrics_dic = {}
for subgroup in feature_subgroups:
    metrics_dic[subgroup] = metrics[subgroup].get_results()
    
countries = metrics_dic.keys()
metrics = list(metrics_dic[feature_subgroups[0]].keys())

keys = list(metrics_dic.keys())
metrics = list(metrics_dic[keys[0]].keys())

plt.figure(figsize=(12, 8))  # Adjust size here
bar_width = 0.07

num_metrics = len(metrics)
index = range(len(keys))

fig, ax = plt.subplots()
for i in range(num_metrics):
    metric_values = [metrics_dic[key][metrics[i]] for key in keys]
    ax.bar([x + i * bar_width for x in index], metric_values, bar_width, label=metrics[i])

ax.set_xlabel('Keys')
ax.set_ylabel('Metrics')
ax.set_title(f'Metrics for Different Keys, trained with k = {user_model_k}')
ax.set_xticks([x + 0.3 for x in index])
ax.set_xticklabels(keys)
ax.legend()

plt.tight_layout()
plt.savefig(f'{figs_path}/Metrics for Different Keys, trained with k = {user_model_k}.pdf')
plt.show()

feature_column = 'user_gender'
feature_subgroups = ['m', 'f']

def plot_evaluate_user_subgroups(model, feature_column = 'user_gender', feature_subgroups = ['m', 'f']):
    _, metrics = evaluate_for_user_subgroups(model)
    metrics_dic = {}
    for subgroup in feature_subgroups:
        metrics_dic[subgroup] = metrics[subgroup].get_results()
        
    countries = metrics_dic.keys()
    metrics = list(metrics_dic[feature_subgroups[0]].keys())

    keys = list(metrics_dic.keys())
    metrics = list(metrics_dic[keys[0]].keys())

    plt.figure(figsize=(12, 8))  # Adjust size here
    bar_width = 0.07

    num_metrics = len(metrics)
    index = range(len(keys))

    _, ax = plt.subplots()
    for i in range(num_metrics):
        metric_values = [metrics_dic[key][metrics[i]] for key in keys]
        ax.bar([x + i * bar_width for x in index], metric_values, bar_width, label=metrics[i])

    ax.set_xlabel('Keys')
    ax.set_ylabel('Metrics')
    ax.set_title(f'Metrics for Different Keys, trained with k = {user_model_k}')
    ax.set_xticks([x + 0.3 for x in index])
    ax.set_xticklabels(keys)
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    plot_evaluate_user_subgroups(model=model, feature_column = 'user_gender', feature_subgroups = ['m', 'f'])
    
user_prototype_model = model.user_feature_extractor.model_1
item_prototype_model = model.item_feature_extractor.model_1



user_embeddings = user_prototype_model.embedding_ext.embedding_layer.weight[:]
user_prototypes = user_prototype_model.prototypes
tsne_plot(objects=user_embeddings.detach().numpy(), prototypes=user_prototypes.detach().numpy(), path_save_fig=f'{figs_path}/user_embedding.pdf')



user_indices = test_data_user_item_augmented['user_id'].values # sample the users you want to plot
# user_indices = user_indices[:100]

attributes_dict = {'user_gender':[], 'user_age':[], 'user_country':[], 'user_num_interactions':[]}
test_data_user_item_augmented_arr = test_data_user_item_augmented[test_data_user_item_augmented['user_id'].isin(user_indices)][list(attributes_dict.keys())].values


# indices
user_indices = user_indices.astype(int)
for i, key in enumerate(attributes_dict.keys()):
    attributes_dict[key] = test_data_user_item_augmented_arr[:, i]


user_embeddings = user_prototype_model.embedding_ext.embedding_layer.weight[user_indices]
user_prototypes = user_prototype_model.prototypes

tsne_plot(objects=user_embeddings.detach().numpy(), prototypes=user_prototypes.detach().numpy(), labels=attributes_dict['user_gender'], title='user_gender', path_save_fig=f'{figs_path}/user_gender_embedding.pdf')
tsne_plot(objects=user_embeddings.detach().numpy(), prototypes=user_prototypes.detach().numpy(), sizes=attributes_dict['user_age'].astype(float)**2, title='User Age', path_save_fig=f'{figs_path}/user_age_embedding.pdf')
tsne_plot(objects=user_embeddings.detach().numpy(), prototypes=user_prototypes.detach().numpy(), sizes=attributes_dict['user_num_interactions'].astype(float), title='User Interactions', path_save_fig=f'{figs_path}/user_interactions_embedding.pdf')
country_counts = test_data_user_item_augmented.groupby(['user_country']).size().sort_values(ascending=False)
chosen_countries = list(country_counts[:3].index) + list(country_counts[40:43].index)

# keep only the rows with the chosen countries
test_data_user_item_augmented_chosen_countries = test_data_user_item_augmented[test_data_user_item_augmented['user_country'].isin(chosen_countries)]
user_countries = test_data_user_item_augmented_chosen_countries['user_country'].values

user_indices = test_data_user_item_augmented_chosen_countries['user_id'].values
user_embeddings = user_prototype_model.embedding_ext.embedding_layer.weight[user_indices]

tsne_plot(objects=user_embeddings.detach().numpy(), prototypes=user_prototypes.detach().numpy(), labels=user_countries, title='User Countries', path_save_fig=f'{figs_path}/user_country_embedding.pdf')

# tracks


item_indices = test_data_user_item_augmented['item_id'].values.astype(int) # sample the users you want to plot
# item_indices = item_indices[:100]
attributes_dict = {'artist_gender':[], 'artist_country':[], 'item_num_interactions':[]}
test_data_user_item_augmented_arr = test_data_user_item_augmented[test_data_user_item_augmented['item_id'].isin(item_indices)][list(attributes_dict.keys())].values
for i, key in enumerate(attributes_dict.keys()):
    attributes_dict[key] = test_data_user_item_augmented_arr[:, i]
item_embeddings = item_prototype_model.embedding_ext.embedding_layer.weight[item_indices]
item_prototypes = item_prototype_model.prototypes
tsne_plot(item_embeddings.detach().numpy(), item_prototypes.detach().numpy(), path_save_fig=f'{figs_path}/item_embedding.pdf', title='Item Embedding')
tsne_plot(item_embeddings.detach().numpy(), item_prototypes.detach().numpy(), sizes=attributes_dict['item_num_interactions'].astype(float), path_save_fig=f'{figs_path}/item_interactions_embedding.pdf', title='Item Interactions')

# clean the data, only Female and Male genders
item_indices = test_data_user_item_augmented.loc[test_data_user_item_augmented.artist_gender.isin(['Male', 'Female'])]['item_id'].values
item_embeddings = item_prototype_model.embedding_ext.embedding_layer.weight[item_indices]
item_prototypes = item_prototype_model.prototypes
attributes_dict['artist_gender'] = test_data_user_item_augmented[test_data_user_item_augmented['item_id'].isin(item_indices)]['artist_gender'].values
tsne_plot(objects=item_embeddings.detach().numpy(), prototypes=item_prototypes.detach().numpy(), labels=attributes_dict['artist_gender'], path_save_fig=f'{figs_path}/artist_gender_embedding.pdf', title='artist genders')
country_counts = test_data_user_item_augmented.groupby(['user_country']).size().sort_values(ascending=False)
chosen_countries = list(country_counts[3:15].index)

# clean the data, only Female and Male genders
item_indices = test_data_user_item_augmented.loc[test_data_user_item_augmented.artist_country.isin(chosen_countries)]['item_id'].values
item_embeddings = item_prototype_model.embedding_ext.embedding_layer.weight[item_indices]
item_prototypes = item_prototype_model.prototypes
attributes_dict['artist_country'] = test_data_user_item_augmented[test_data_user_item_augmented['item_id'].isin(item_indices)]['artist_country'].values
tsne_plot(objects=item_embeddings.detach().numpy(), prototypes=item_prototypes.detach().numpy(), labels=attributes_dict['artist_country'], path_save_fig=f'{figs_path}/item_country_embedding.pdf', title='TSNE FOR more COMMON COUNTRIES')
    
    