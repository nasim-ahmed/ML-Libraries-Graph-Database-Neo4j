from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import os
import argparse
import json

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def shuffle_stays(stays, seed=9):
    return shuffle(stays, random_state=seed)

def process_table(table_name, table, stays, folder_path):
    # table = table.reindex(stays)
    table = table.loc[stays].copy()
    table.to_csv('{}/{}.csv'.format(folder_path, table_name))
    return table


def split_train_test(eICU_path, is_test=True, seed=9, cleanup=True):
    labels = pd.read_csv(eICU_path + 'preprocessed_labels.csv')
    labels.set_index('uniquepid', inplace=True)

    train, test = train_test_split(labels.index, test_size=0.15, random_state=seed)
    train, val = train_test_split(train, test_size=0.15/0.85, random_state=seed)
   
    print('==> Loading data for splitting...')
    # if is_test:
    #     timeseries = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv', nrows=999999)
    # else:
    #     timeseries = pd.read_csv(eICU_path + 'preprocessed_timeseries.csv')
    # timeseries.set_index('uniquepid', inplace=True)
    diagnoses = pd.read_csv(eICU_path + 'preprocessed_diagnoses.csv')
    diagnoses.set_index('uniquepid', inplace=True)
    flat_features = pd.read_csv(eICU_path + 'preprocessed_flat.csv')
    flat_features.set_index('uniquepid', inplace=True)

    # delete the source files, as they won't be needed anymore
    # if is_test is False and cleanup:
    #     print('==> Removing the unsorted data...')
    #     os.remove(eICU_path + 'preprocessed_timeseries.csv')
    #     os.remove(eICU_path + 'preprocessed_diagnoses.csv')
    #     os.remove(eICU_path + 'preprocessed_labels.csv')
    #     os.remove(eICU_path + 'preprocessed_flat.csv')

    for partition_name, partition in zip(['train', 'val', 'test'], [train, val, test]):
        print('==> Preparing {} data...'.format(partition_name))
        stays = partition
        folder_path = create_folder(eICU_path, partition_name)
        stays = shuffle_stays(stays, seed=9)
        with open(folder_path + '/stays.txt', 'w') as f:
            for table_name, table in zip(['labels', 'flat', 'diagnoses'],
                                         [labels, flat_features, diagnoses]):
                table = process_table(table_name, table, stays, folder_path)
                for stay in table.index:
                    f.write("%s\n" % stay)
    return

if __name__=='__main__':
    # with open('paths.json', 'r') as f:
    #     eICU_path = json.load(f)["eICU_path"]
    eICU_path = '/media/nasim/31c299f0-f952-4032-9bd8-001b141183e0/ML-Libraries-Graph-Database-Neo4j/PyG-Neo4j/app/eICU_data/'
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    split_train_test(eICU_path, is_test=False, cleanup=args.cleanup)