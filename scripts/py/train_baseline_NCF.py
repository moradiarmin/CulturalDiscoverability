import numpy as np
import pandas as pd
import os 
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import argparse
import wandb
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

# parse args and set paths
parser = argparse.ArgumentParser(description="Argparser for modifying the training settings")
parser.add_argument("--logs_path", type=str, help="Path to save outputs")
parser.add_argument("--dataset_path", type=str, help="Path to read datasets")
args = parser.parse_args()

# DATASET_PATH = args.dataset_path
# DATASET_PATH = '/home/mila/a/armin.moradi/scratch/data/LFM_2b_seperated_final'
DATASET_PATH = '/home/mila/a/armin.moradi/ProtoMF/data/lfm2b-1mon'

# LOGS_PATH = '/home/mila/a/armin.moradi/CulturalDiscoverability/results/model_outputs_testing_ipynb/'

LOGS_PATH = args.logs_path
ITEM_FRACTION = 1.0
USER_FRACTION = 0.01
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_SIZE = [16, 32, 64]
EMBEDDING_SIZE = 8

# wandb
wandb.init(project='cultural-rs', entity='armornine')
wandb.config["model"] = "AgnosticNCF"
wandb.config["user_fraction"] = USER_FRACTION
wandb.config["item_fraction"] = ITEM_FRACTION
wandb.config.update({"batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "hidden_size": HIDDEN_SIZE, "embedding_size": EMBEDDING_SIZE})
wandb.config.update(args)
print('Initialized WandB')
# load data

interactions = pd.read_csv(os.path.join(DATASET_PATH, 'listening_history_train.csv'))
user_demographics = pd.read_csv(os.path.join(DATASET_PATH, 'user_ids.csv'))
item_demographics = pd.read_csv(os.path.join(DATASET_PATH, 'item_ids.csv'))

print('Loaded Data', 'interactions:', len(interactions), 'users', len(user_demographics), 'items', len(item_demographics))
print('-'*50)

interactions['count'] = 1

# reindex IDs

user_demographics['new_user_id'] = user_demographics.index
user_demographics = user_demographics.sample(frac=USER_FRACTION, random_state=99) # TODO: CHANGE TO COMPLETE
print('Sampled User Demographics:', len(user_demographics))
      
df_final = interactions.merge(user_demographics, on='user_id', how='inner')

print(len(df_final), 'len of sampled user interactions')
# add random negative samples

user_ids = df_final['user_id'].unique()
item_ids = interactions['track_id'].unique()

negative_samples = []
for user_id in user_ids:
    user_interacted_items = interactions[interactions['user_id'] == user_id]['track_id'].values
    user_negative_items = np.setdiff1d(item_ids, user_interacted_items)
    user_negative_items = np.random.choice(user_negative_items, size=len(user_interacted_items), replace=False)
    for negative_item in user_negative_items:
        negative_samples.append([user_id, negative_item, 0])

negative_interactions = pd.DataFrame(negative_samples, columns=['user_id', 'track_id', 'count'])
negative_df = negative_interactions.merge(user_demographics, on='user_id', how='inner')
df_final = pd.concat([df_final, negative_df], ignore_index=True)

print('Added Negative Samples', '~ Added Items:', len(df_final[df_final['count'] == 0]))
print('-'*50)

item_demographics['new_track_id'] = item_demographics.index
item_demographics = item_demographics[item_demographics['track_id'].isin(interactions['track_id'])]
item_demographics = item_demographics.sample(frac=ITEM_FRACTION, random_state=99) #TODO: CHANGE TO COMPLETE
print('Sampled Item Demographics:', len(item_demographics))

df_final = df_final.merge(item_demographics, on='track_id', how='inner')

print('DF FINAL - n rows:', len(df_final))
print('-'*50)

# preprocessing

labels = ['0-18', '18-30', '30-50', '50+']
bins = [0, 18, 30, 50, 100]
df_final['user_age'] = pd.cut(df_final['user_age'], bins=bins, labels=labels, right=False)

# replace user id with new_user_id and delete user_id
df_final = df_final.drop(columns=['user_id'])
df_final = df_final.rename(columns={'new_user_id':'user_id'})

# replace user id with new_user_id and delete user_id
df_final = df_final.drop(columns=['track_id'])
df_final = df_final.rename(columns={'new_track_id':'track_id'})

cols = df_final.columns.tolist()
cols = [cols[-1], cols[3]] + cols[0:3] + cols[4:-1]
df_final = df_final[cols]
print('Finished Pre-processing')
print('-'*50)


# shape data

y = df_final['count']
x = df_final.drop(['count'], axis=1)
encoder = OneHotEncoder(sparse=False)
columns_to_encode = x.columns[2:]
encoder.fit(x[columns_to_encode])
encoded_x = encoder.transform(x[columns_to_encode])
encoded_x = pd.DataFrame(encoded_x, columns=encoder.get_feature_names_out(columns_to_encode))
x = pd.concat([x[['user_id', 'track_id']], encoded_x], axis=1)

user_country_indices = [i for i, col in enumerate(x.columns) if col.startswith('user_country_')]
artist_country_indices = [i for i, col in enumerate(x.columns) if col.startswith('artist_country_')]
user_gender_indices = [i for i, col in enumerate(x.columns) if col.startswith('user_gender_')]
artist_gender_indices = [i for i, col in enumerate(x.columns) if col.startswith('artist_gender')]
user_age_indices = [i for i, col in enumerate(x.columns) if col.startswith('user_age_')]
unique_user_ids, unique_track_ids = max(x['user_id']) + 1, max(x['track_id']) + 1

# split the interactions of each user into train, val, and test (80-10-10)

train_share = 0.75
val_share = 0.15
test_share = 1 - train_share - val_share

user_interactions = {}
for i in range(len(x)):
    user_id = x.iloc[i]['user_id']
    if user_id not in user_interactions:
        user_interactions[user_id] = []
    user_interactions[user_id].append(i)

train_indices, val_indices, test_indices = [], [], []
for user_id, indices in user_interactions.items():
    np.random.shuffle(indices)
    train_indices += indices[:int(train_share * len(indices))]
    val_indices += indices[int(train_share * len(indices)):int((train_share + val_share ) * len(indices))]
    test_indices += indices[int((1 - test_share) * len(indices)):]
    
train_x, train_y = x.iloc[train_indices], y.iloc[train_indices]
val_x, val_y = x.iloc[val_indices], y.iloc[val_indices]
test_x, test_y = x.iloc[test_indices], y.iloc[test_indices]

# create dataloaders

train_dataset = TensorDataset(torch.tensor(train_x.values, dtype=torch.float), torch.tensor(train_y.values, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(val_x.values, dtype=torch.float), torch.tensor(val_y.values, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(test_x.values, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f'created loaders with batchsize {BATCH_SIZE}', len(train_dataset), len(val_dataset), len(test_dataset))

# training model with only user_id and track_id and countries

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_tracks, user_country_dim, artist_country_dim, hidden_size=HIDDEN_SIZE):
        super(NeuralCollaborativeFiltering, self).__init__()

        self.embedding_size = EMBEDDING_SIZE
        self.user_id_embedding = nn.Embedding(num_users, self.embedding_size)
        self.track_id_embedding = nn.Embedding(num_tracks, self.embedding_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.embedding_size * 2 + user_country_dim + artist_country_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 1),
        )

    def forward(self, user_id, artist_id, user_country, artist_country):

        user_id_embeds = self.user_id_embedding(user_id)
        track_id_embeds = self.track_id_embedding(artist_id)
        concatenated = torch.cat([user_id_embeds, track_id_embeds, user_country, artist_country], dim=1)
        output = self.fc_layers(concatenated.float())

        return output.squeeze()


class AgnosticNCF(nn.Module):
    def __init__(self, num_users, num_tracks, user_country_dim, artist_country_dim, hidden_size=[16, 32, 32]): # 256, 128, 64
        super(AgnosticNCF, self).__init__()

        self.embedding_size = 8 # 32

        self.user_id_embedding = nn.Embedding(num_users, self.embedding_size)
        self.track_id_embedding = nn.Embedding(num_tracks, self.embedding_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.embedding_size * 2, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            nn.Linear(hidden_size[2], 1),
        )

    def forward(self, user_id, artist_id, user_country, artist_country):

        user_id_embeds = self.user_id_embedding(user_id)
        track_id_embeds = self.track_id_embedding(artist_id)
        

        concatenated = torch.cat([user_id_embeds, track_id_embeds], dim=1)
        output = self.fc_layers(concatenated.float())

        return output.squeeze()

print('Start Training')
print('-'*50)

start_time = time.time()

# model = NeuralCollaborativeFiltering(num_users=unique_user_ids, num_tracks=unique_track_ids,
#                                     user_country_dim=len(user_country_indices),
#                                     artist_country_dim=len(artist_country_indices),
#                                     hidden_size=[64, 128, 64]).to(device)

model = AgnosticNCF(num_users=unique_user_ids, num_tracks=unique_track_ids,
                                    user_country_dim=len(user_country_indices),
                                    artist_country_dim=len(artist_country_indices),
                                    hidden_size=[64, 128, 64]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

val_loss = []
train_loss = []


for epoch in tqdm(range(EPOCHS), desc='Epochs'):
    model.train()
    total_loss = 0.0
    for xx, yy in train_loader:
        user_id = xx[:, 0].long().to(device)
        track_id = xx[:, 1].long().to(device)

        user_country = xx[:, user_country_indices].long().to(device)
        artist_country = xx[:, artist_country_indices].long().to(device)
        
        optimizer.zero_grad()
        outputs = model(user_id, track_id, user_country, artist_country)
        loss = criterion(outputs.float(), yy.float().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    wandb.log({'train_loss': total_loss / len(train_loader)})
    
    model.eval()
    total_val_loss = 0.0
    for xx, yy in val_loader:
        user_id = xx[:, 0].long().to(device)
        track_id = xx[:, 1].long().to(device)

        user_country = xx[:, user_country_indices].long().to(device)
        artist_country = xx[:, artist_country_indices].long().to(device)

        outputs = model(user_id, track_id, user_country, artist_country)
        loss = criterion(outputs.float(), yy.float().to(device))
        total_val_loss += loss.item()
    wandb.log({'val_loss': total_val_loss / len(val_loader)})
        
    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}, Val Loss: {total_val_loss / len(val_loader)}')
    train_loss.append(total_loss / len(train_loader))
    val_loss.append(total_val_loss / len(val_loader))

    torch.save(model.state_dict(), LOGS_PATH + 'model_weights.pth')

print('Training Time:', round((time.time() - start_time)/60, 2), 'minutes')
print('-'*50)

# test

model.eval()
total_test_loss = 0.0
diff_list = []

with torch.no_grad():
    for xx, yy in test_loader:
        user_id = xx[:, 0].long().to(device)
        track_id = xx[:, 1].long().to(device)

        user_country = xx[:, user_country_indices].long().to(device)
        artist_country = xx[:, artist_country_indices].long().to(device)

        outputs = model(user_id, track_id, user_country, artist_country)
        loss = criterion(outputs.float(), yy.float().to(device))
        total_test_loss += loss.item()
        diff = abs(outputs - yy.float().to(device))
        diff_list.append(diff)
        wandb.log({'diff mean': diff.mean().item()})
        wandb.log({'diff std': diff.std().item()})
    wandb.log({'test_loss': total_test_loss / len(test_loader)})
    
mean_diff = torch.cat(diff_list).mean().item()
std_diff = torch.cat(diff_list).std().item()

print(f'Test Loss: {total_test_loss / len(test_loader)}')
print(f'Mean Difference: {mean_diff}')
print(f'Standard Deviation of Difference: {std_diff}')

wandb.finish()