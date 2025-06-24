import os
import pandas as pd

YEAR_MIN = 2011
YEAR_MAX = 2019

SRC_DIR = 'dataset2'
DEST_DIR = 'dataset2_filtered'

splits = ['train', 'valid', 'test']

os.makedirs(DEST_DIR, exist_ok=True)

for split in splits:
    src_split = os.path.join(SRC_DIR, split)
    dst_split = os.path.join(DEST_DIR, split)
    os.makedirs(dst_split, exist_ok=True)

    # load all_infos
    info_path = os.path.join(src_split, f'{split}_all_infos.xlsx')
    df_info = pd.read_excel(info_path)
    df_info['year'] = df_info['cve_id'].str.extract(r'CVE-(\d{4})-')[0].astype(int)
    mask = (df_info['year'] < YEAR_MIN) | (df_info['year'] > YEAR_MAX)
    df_filtered = df_info[mask].drop(columns=['year'])

    df_filtered.to_excel(os.path.join(dst_split, f'{split}_all_infos.xlsx'), index=False)

    idx = df_filtered.index

    # helper to filter csvs by index
    def filter_csv(name):
        path = os.path.join(src_split, name)
        if not os.path.exists(path):
            return
        df = pd.read_csv(path, header=None)
        df_filtered_csv = df.loc[idx]
        df_filtered_csv.to_csv(os.path.join(dst_split, name), index=False, header=False)

    filter_csv(f'{split}_code.csv')
    filter_csv(f'{split}_ast.csv')
    filter_csv(f'{split}_desc.csv')
    filter_csv(f'{split}_bscore.csv')
    filter_csv(f'{split}_bseveritys.csv')

    # also copy xlsx version if exists
    all_xlsx = os.path.join(src_split, f'{split}_all.xlsx')
    if os.path.exists(all_xlsx):
        df = pd.read_excel(all_xlsx)
        df_filtered_xlsx = df.loc[idx]
        df_filtered_xlsx.to_excel(os.path.join(dst_split, f'{split}_all.xlsx'), index=False)
