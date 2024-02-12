# lfm-2b sample merge tracks with mb

import pandas as pd
import os


tracks = '/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/data/lfm2b-1mon/tracks.tsv'
tracks_df = pd.read_csv(tracks, sep='\t')
artists_df = tracks_df[['artist']].drop_duplicates()
print('Loaded LFM-2b')
scratch_directory = os.environ.get('SCRATCH') + '/'
MB_DATASETS = ["artist", "release", "release-group"]
MB_ROOT_DIR = scratch_directory + "/data/json-dumps/20231028-001001/{0}/mbdump/{0}"
artists_mb = pd.read_json(MB_ROOT_DIR.format("artist"), lines=True, chunksize=100000)
print('Loaded MB artists')

DST_FOLDER = '/home/mila/a/armin.moradi/CulturalDiscoverability/ProtoMF/data/lfm2b-1mon/artists_augmented.csv'

i = 0
for artist_mb_chunk in artists_mb:
    if i == 0:
        result_df = pd.merge(artist_mb_chunk, artists_df, left_on='name', right_on='artist', how='inner')
        result_df.to_csv(DST_FOLDER, mode='w', index=False)    
        print(i, 'Done!')
        i += 1
    if i > 0:
        result_df = pd.merge(artist_mb_chunk, artists_df, left_on='name', right_on='artist', how='inner')
        result_df.to_csv(DST_FOLDER, mode='a', header=False, index=False)    
        print(i, 'Done!')
        i += 1