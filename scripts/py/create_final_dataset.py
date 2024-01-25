import pandas as pd
import os

scratch_directory = os.environ.get('SCRATCH')
home_directory = os.environ.get('HOME')

DATASET_DIR = scratch_directory + '/data/LFM-2b/'
DST_DIR = scratch_directory + '/data/{0}'

users_df = pd.read_csv(DATASET_DIR + 'users.tsv.bz2', compression='bz2', error_bad_lines=False, sep='\t')
tracks_df = pd.read_csv(DATASET_DIR + 'tracks.tsv.bz2', compression='bz2', error_bad_lines=False, sep='\t')
listening_counts_df = pd.read_csv(DATASET_DIR + 'listening-counts.tsv.bz2', compression='bz2', error_bad_lines=False, sep='\t')
artists_df = pd.read_csv(DATASET_DIR + 'merged_artists.csv', error_bad_lines=False)

users_df = users_df.rename(columns={'country': 'user_country', 'age': 'user_age', 'gender': 'user_gender', 'creation_time': 'user_creation_time'})
artists_df = artists_df.rename(columns={'country': 'artist_country', 'gender': 'artist_gender', 'area': 'artist_area'})
print(artists_df.columns, users_df.columns, tracks_df.columns, listening_counts_df.columns)

tracks_artists_df = tracks_df.merge(artists_df[['artist_name', 'artist_gender', 'artist_country']], on='artist_name')
print('Merged tracks and artists', len(tracks_artists_df))

tracks_artists_interactions_df = tracks_artists_df.merge(listening_counts_df, on='track_id')
print('Merged with events!', len(tracks_artists_interactions_df))

all_interactions_df = tracks_artists_interactions_df.merge(users_df, on='user_id')
print('Merged with users!', len(all_interactions_df))



DST_FOLDER = DST_DIR.format('final_lfm_2b_dataset')
all_interactions_df.to_csv(DST_FOLDER, mode='w', index=False)
