import pandas as pd
import os

scratch_directory = os.environ.get('SCRATCH')
home_directory = os.environ.get('HOME')

MB_DATASETS = ["artist", "release", "release-group"]
MB_ROOT_DIR = scratch_directory + "/data/json-dumps/20231028-001001/{0}/mbdump/{0}"
LFM_DATASETS = ["albums", "artists", "listening-counts", "listening-events", "lyrics-features", "spotify-uris", "tags-micro-genres", "tags", "tracks", "users"]
FORMAT = ["tsv", "json"]
LFM_ROOT_DIR = scratch_directory + "/data/LFM-2b/{0}.{1}.bz2"

artists_lfm = pd.read_csv(LFM_ROOT_DIR.format("artists", "tsv"), sep="\t")
artists_mb = pd.read_json(MB_ROOT_DIR.format("artist"), lines=True, chunksize=100000)

i = 0
DST_FOLDER = f"{scratch_directory}/data/merged_artists.csv"
for artist_mb_chunk in artists_mb:
    result_df = pd.merge(artist_mb_chunk, artists_lfm, left_on='name', right_on='artist_name', how='inner')
    if i == 0:
        result_df.to_csv(DST_FOLDER, mode='w', index=False)
        print(i, 'Done!')
        i+= 1
    else:
        result_df.to_csv(DST_FOLDER, mode='a', header=False, index=False)
        i += 1
        print(i, 'Done!')
