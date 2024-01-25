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

# wandb
wandb.init(project='cultural-rs', entity='armornine')
print('Initialized Wandb')

# parse args

parser = argparse.ArgumentParser(description="Argparser for modifying the training settings")
parser.add_argument("--logs_path", type=str, help="Path to save outputs")
parser.add_argument("--dataset_path", type=str, help="Path to read datasets")

args = parser.parse_args()

DATASET_PATH = args.dataset_path
LOGS_PATH = args.logs_path

print('Parsed Arguments')
print('DATASET_PATH:', DATASET_PATH)
print('LOGS_PATH:', LOGS_PATH)
print('-'*50)

# load data

interactions = pd.read_csv(os.path.join(DATASET_PATH, '10k_sampled_interactions.csv'))
user_demographics = pd.read_csv(os.path.join(DATASET_PATH, '10k_sampled_users.csv'))
item_demographics = pd.read_csv(os.path.join(DATASET_PATH, 'item_demographics.csv'))

print('Loaded Data', 'interactions:', len(interactions), 'users', len(user_demographics), 'items', len(item_demographics))
print('-'*50)

# reindex IDs
user_demographics['new_user_id'] = user_demographics.index
user_demographics = user_demographics.sample(n=250, random_state=99) # TODO: CHANGE TO COMPLETE
df_final = interactions.merge(user_demographics, on='user_id', how='inner')

item_demographics['new_track_id'] = item_demographics.index
item_demographics = item_demographics[item_demographics['track_id'].isin(interactions['track_id'])]
item_demographics = item_demographics.sample(frac=1, random_state=99) #TODO: CHANGE TO COMPLETE

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

print('Finished Pre-processing, DF FINAL')
print(df_final.iloc[0])
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

x_tensor = torch.tensor(x.values)
y_tensor = torch.tensor(y.values)
dataset = TensorDataset(x_tensor, y_tensor)

train_ratio = 0.75
val_ratio = 0.10
test_ratio = 0.15

num_samples = len(dataset)
num_train_samples = int(train_ratio * num_samples)
num_val_samples = int(val_ratio * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

print('num_samples:', num_samples)
print('num_train_samples:', num_train_samples)
print('num_val_samples:', num_val_samples)
print('num_test_samples:', num_test_samples)

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [num_train_samples, num_val_samples, num_test_samples])

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

print('Finished Making the Dataloaders')
print('-'*50)

# training model with only user_id and track_id and countries

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_tracks, user_country_dim, artist_country_dim, hidden_size=[256, 128, 64]):
        super(NeuralCollaborativeFiltering, self).__init__()

        self.embedding_size = 16

        self.user_id_embedding = nn.Embedding(num_users, self.embedding_size)
        self.track_id_embedding = nn.Embedding(num_tracks, self.embedding_size)

        # self.user_country_embedding = nn.Embedding(user_country_dim, self.embedding_size)
        # self.artist_country_embedding = nn.Embedding(artist_country_dim, self.embedding_size)

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
        artist_id_embeds = self.track_id_embedding(artist_id)
        
        # user_country_embeds = self.user_country_embedding(user_country)
        # artist_country_embeds = self.artist_country_embedding(artist_country)

        concatenated = torch.cat([user_id_embeds, artist_id_embeds, user_country, artist_country], dim=1)
        output = self.fc_layers(concatenated.float())

        return output.squeeze()

print('Start Training\n')
print('-'*50)


start_time = time.time()

model = NeuralCollaborativeFiltering(num_users=unique_user_ids, num_tracks=unique_track_ids,
                                    user_country_dim=len(user_country_indices),
                                    artist_country_dim=len(artist_country_indices),
                                    hidden_size=[256, 128, 64]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

val_loss = []
train_loss = []


for epoch in tqdm(range(num_epochs), desc='Epochs'):
    
    model.train()
    total_loss = 0.0
    for xx, yy in train_loader:
        user_id = xx[:, 0].long().to(device)
        artist_id = xx[:, 1].long().to(device)

        user_country = xx[:, user_country_indices].long().to(device)
        artist_country = xx[:, artist_country_indices].long().to(device)
        
        optimizer.zero_grad()
        outputs = model(user_id, artist_id, user_country, artist_country)
        loss = criterion(outputs.float(), yy.float().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    wandb.log({'train_loss': total_loss / len(train_loader)})
    
    model.eval()
    total_val_loss = 0.0
    for xx, yy in val_loader:
        user_id = xx[:, 0].long().to(device)
        artist_id = xx[:, 1].long().to(device)

        user_country = xx[:, user_country_indices].long().to(device)
        artist_country = xx[:, artist_country_indices].long().to(device)

        outputs = model(user_id, artist_id, user_country, artist_country)
        loss = criterion(outputs.float(), yy.float().to(device))
        total_val_loss += loss.item()
    wandb.log({'val_loss': total_val_loss / len(val_loader)})
        
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}, Val Loss: {total_val_loss / len(val_loader)}')
    train_loss.append(total_loss / len(train_loader))
    val_loss.append(total_val_loss / len(val_loader))

torch.save(model.state_dict(), LOGS_PATH + 'model_weights.pth')
print('Saved Trained Model')
print('Training Time:', time.time() - start_time)
print('-'*50)

# test

model.eval()
total_test_loss = 0.0
diff_list = []

with torch.no_grad():
    for xx, yy in test_loader:
        user_id = xx[:, 0].long().to(device)
        artist_id = xx[:, 1].long().to(device)

        user_country = xx[:, user_country_indices].long().to(device)
        artist_country = xx[:, artist_country_indices].long().to(device)

        outputs = model(user_id, artist_id, user_country, artist_country)
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

# np.save(LOGS_PATH + 'test_loss.npy', np.array(test_loss))
# torch.save(diff_list, LOGS_PATH + 'diff_list.pt')

wandb.finish()