import warnings
warnings.filterwarnings("ignore")

import gc
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':

    # Read log datas and file_id datas
    dir_path = './'
    train_set = pd.read_csv(dir_path+'training-set.csv', header=None, names=['FileID', 'Target'])
    test_set = pd.read_csv(dir_path+'testing-set.csv', header=None, names=['FileID', 'Target'])
    train_ex = pd.read_table(dir_path+'exception/exception_train.txt', header=None, names=['FileID'])
    test_ex = pd.read_table(dir_path+'exception/exception_testing.txt', header=None, names=['FileID'])

    train_set = train_set.loc[~train_set['FileID'].isin(train_ex)]
    test_set = test_set.loc[~test_set['FileID'].isin(test_ex)]
    all_set = pd.concat([train_set['FileID'], test_set['FileID']])
    print('Load FileID complete.')

    log_data = pd.DataFrame(data=[], columns=['FileID', 'CustomerID', 'QueryTS', 'ProductID'])
    for file in glob.glob(dir_path+'query_log/*.csv'):
        tmp = pd.read_csv(file, header=None, names=['FileID', 'CustomerID', 'QueryTS', 'ProductID'])
        tmp['QueryTS'] = pd.to_datetime(tmp['QueryTS'],unit='s')
        log_data = pd.concat([log_data, tmp.loc[tmp['FileID'].isin(all_set)]], axis=0)
    print('Combine Logs complete.')
    
    # Sorting log_data by QueryTS
    log_data['QueryTS'] = pd.to_datetime(log_data['QueryTS'], format='%Y-%m-%d %H:%M:%S')
    log_data.sort_values(['QueryTS'], ascending=True, na_position='first', inplace=True)
    log_data.reset_index(drop=True, inplace=True)
    log_data['ProductID'] = log_data['ProductID'].astype(str)
    print('Sorting data complete.')
 
    # Generate data set by concat train_set and test_set
    data = pd.concat([train_set, test_set],axis=0)
    data['Target'].replace(0.5, np.nan, inplace=True)
    nrow_train = len(train_set)

    del train_ex, test_ex
    gc.collect()

    # Fix fold number for each FileID due to prevent overfitting
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, random_state=5566)
    data['Fold'] = np.nan
    for f, (_, valid_idx) in enumerate(skf.split(data['FileID'].iloc[:nrow_train], data['Target'].iloc[:nrow_train])):
        data['Fold'].iloc[valid_idx] = f
    log_data = pd.merge(log_data, data, on='FileID', how='left')    


    log_data.to_csv(dir_path+'log_data.csv', index=False)
    data.to_csv(dir_path+'data.csv', index=False)
    print('Write data into csv complete.')

    del tmp, all_set, file
    gc.collect()
