import numpy as np
import pandas as pd
import os 
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import wandb
import time
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = '/home/mila/a/armin.moradi/ProtoMF/data/lfm2b-1mon'
ITEM_FRACTION = 1.0
USER_FRACTION = 1.0
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100 # TODO: TO CHANGe
HIDDEN_SIZE = [16, 32, 64]
EMBEDDING_SIZE = 8


parser = argparse.ArgumentParser(description="Argparser for modifying the training settings")
parser.add_argument("--logs_path", type=str, help="Path to save outputs")
args = parser.parse_args()

LOGS_PATH = args.logs_path
DATASET_PATH = '/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/data/lfm2b-1mon'
interactions = pd.read_csv(os.path.join(DATASET_PATH, 'listening_history_train.csv'))
user_demographics = pd.read_csv(os.path.join(DATASET_PATH, 'user_ids.csv'))
item_demographics = pd.read_csv(os.path.join(DATASET_PATH, 'item_ids.csv'))

print('Loaded Data', 'interactions:', len(interactions), 'users', len(user_demographics), 'items', len(item_demographics))
print('-'*50)

interactions = interactions[['user_id', 'item_id']]
interactions['count'] = 1
interactions.head()

# take only 10 interactions
# interactions = interactions.sample(frac=0.01, random_state=42) #TODO REMOVE

print('Sampled Interactions', len(interactions))
df_final = interactions

user_ids = df_final['user_id'].unique()
item_ids = interactions['item_id'].unique()

negative_samples = []
for user_id in user_ids:
    user_interacted_items = interactions[interactions['user_id'] == user_id]['item_id'].values
    user_negative_items = np.setdiff1d(item_ids, user_interacted_items)
    user_negative_items = np.random.choice(user_negative_items, size=len(user_interacted_items), replace=False)
    for negative_item in user_negative_items:
        negative_samples.append([user_id, negative_item, 0])

negative_interactions = pd.DataFrame(negative_samples, columns=['user_id', 'item_id', 'count'])
# negative_df = negative_interactions.merge(user_demographics, on='user_id', how='inner')
df_final = pd.concat([df_final, negative_interactions], ignore_index=True)

print('Added Negative Samples', '~ Added Items:', len(df_final[df_final['count'] == 0]))
print('-'*50)

 # drop old_user_id 
# df_final = df_final.drop(columns=['old_user_id'])
df_final.head()

# add up the columns with repeated user_id and item_id and count
df_final = df_final.groupby(['user_id', 'item_id']).sum().reset_index()
print('FINAL DF', df_final.head())

y = df_final['count']
x = df_final.drop(['count'], axis=1)
encoder = OneHotEncoder()
# columns_to_encode = x.columns[2:]
# encoder.fit(x[columns_to_encode])
# encoded_x = encoder.transform(x[columns_to_encode])
# encoded_x = pd.DataFrame(encoded_x, columns=encoder.get_feature_names_out(columns_to_encode))
# x = pd.concat([x[['user_id', 'track_id']], encoded_x], axis=1)

# user_country_indices = [i for i, col in enumerate(x.columns) if col.startswith('user_country_')]
# artist_country_indices = [i for i, col in enumerate(x.columns) if col.startswith('artist_country_')]
# user_gender_indices = [i for i, col in enumerate(x.columns) if col.startswith('user_gender_')]
# artist_gender_indices = [i for i, col in enumerate(x.columns) if col.startswith('artist_gender')]
# user_age_indices = [i for i, col in enumerate(x.columns) if col.startswith('user_age_')]
# unique_user_ids, unique_track_ids = max(x['user_id']) + 1, max(x['track_id']) + 1

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


train_x.head()


# create dataloaders

train_dataset = TensorDataset(torch.tensor(train_x.values, dtype=torch.float), torch.tensor(train_y.values, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(val_x.values, dtype=torch.float), torch.tensor(val_y.values, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(test_x.values, dtype=torch.float), torch.tensor(test_y.values, dtype=torch.float))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f'created loaders with batchsize {BATCH_SIZE}', len(train_dataset), len(val_dataset), len(test_dataset))

# set up wandb logging

wandb.init(project='fooling-around', entity='armornine')
print('Initialized WandB')

unique_user_ids = df_final['user_id'].nunique()
unique_track_ids = df_final['item_id'].nunique()

class AgnosticNCF(nn.Module):
    def __init__(self, num_users, num_tracks, hidden_size=[16, 32, 32]): # 256, 128, 64
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

    def forward(self, user_id, artist_id):

        user_id_embeds = self.user_id_embedding(user_id)
        track_id_embeds = self.track_id_embedding(artist_id)
        

        concatenated = torch.cat([user_id_embeds, track_id_embeds], dim=1)
        output = self.fc_layers(concatenated.float())

        return output.squeeze()

print('Start Training')
print('-'*50)

start_time = time.time()
model = AgnosticNCF(num_users=unique_user_ids, num_tracks=unique_track_ids, hidden_size=[64, 128, 64]).to(device)

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

        # user_country = xx[:, user_country_indices].long().to(device)
        # artist_country = xx[:, artist_country_indices].long().to(device)

        optimizer.zero_grad()
        outputs = model(user_id, track_id)# , user_country, artist_country)
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

        # user_country = xx[:, user_country_indices].long().to(device)
        # artist_country = xx[:, artist_country_indices].long().to(device)

        outputs = model(user_id, track_id)
        loss = criterion(outputs.float(), yy.float().to(device))
        total_val_loss += loss.item()
    wandb.log({'val_loss': total_val_loss / len(val_loader)})
        
    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}, Val Loss: {total_val_loss / len(val_loader)}')
    train_loss.append(total_loss / len(train_loader))
    val_loss.append(total_val_loss / len(val_loader))

    if epoch % 5 == 1:
        torch.save(model.state_dict(), LOGS_PATH + f'ncf_model_epoch_{epoch}.pt')

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

        # user_country = xx[:, user_country_indices].long().to(device)
        # artist_country = xx[:, artist_country_indices].long().to(device)

        outputs = model(user_id, track_id)
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