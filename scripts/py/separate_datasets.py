import os
import pandas as pd

df_chunks = pd.read_csv('/home/mila/a/armin.moradi/scratch/data/final_lfm_2b_dataset', chunksize=1000000, low_memory=False)

i = 0
for df in df_chunks:
    print('Read Done!', len(df))

    user_demographics = df[['user_id', 'user_country', 'user_gender', 'user_age']] 
    item_demographics = df[['track_id', 'user_id', 'artist_gender', 'artist_country']]  
    interactions = df[['user_id', 'track_id', 'count']]

    user_demographics = user_demographics.drop_duplicates(subset='user_id')
    user_demographics = user_demographics[user_demographics['user_country'].str.len() < 3]
    median_age = user_demographics[user_demographics['user_age'] > 0]['user_age'].median()
    user_demographics['user_age'] = user_demographics['user_age'].apply(lambda x: median_age if x <= 0 else x)
    user_demographics['user_age'] = user_demographics['user_age'].fillna(user_demographics['user_age'].median())
    print('Users!', len(user_demographics))

    item_demographics = item_demographics.drop_duplicates(subset='track_id')
    item_demographics = item_demographics[item_demographics['user_id'].isin(user_demographics['user_id'])]
    item_demographics = item_demographics[item_demographics['artist_country'].str.len() < 3]
    item_demographics = item_demographics.loc[(item_demographics['artist_gender'] == 'Male') | (item_demographics['artist_gender'] == 'Female')]
    item_demographics['artist_gender'] = item_demographics['artist_gender'].replace('Female', 'f')
    item_demographics['artist_gender'] = item_demographics['artist_gender'].replace('Male', 'm')
    item_demographics = item_demographics.drop(columns='user_id')
    print('Done Item!', len(item_demographics))

    interactions = interactions[interactions['user_id'].isin(user_demographics['user_id'])]
    interactions = interactions[interactions['track_id'].isin(item_demographics['track_id'])]
    interactions = interactions.drop_duplicates(subset=['user_id', 'track_id'])
    print('Done Interactions!', len(interactions))


    PATH = '/home/mila/a/armin.moradi/scratch/data/LFM_2b_seperated_final/'

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    if i == 0:
        interactions.to_csv(os.path.join(PATH, 'interactions.csv'), index=False, mode='w')
        user_demographics.to_csv(os.path.join(PATH, 'user_demographics.csv'), index=False, mode='w')
        item_demographics.to_csv(os.path.join(PATH, 'item_demographics.csv'), index=False, mode='w')
        print('WROTE:', len(interactions), len(user_demographics), len(item_demographics))
    else:
        interactions.to_csv(os.path.join(PATH, 'interactions.csv'), index=False, mode='a', header=False)
        user_demographics.to_csv(os.path.join(PATH, 'user_demographics.csv'), index=False, mode='a', header=False)
        item_demographics.to_csv(os.path.join(PATH, 'item_demographics.csv'), index=False, mode='a', header=False)
        print('WROTE:', len(interactions), len(user_demographics), len(item_demographics))

    i += 1