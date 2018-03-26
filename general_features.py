import warnings
warnings.filterwarnings("ignore")

import re
import gc
import datetime
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import implicit

import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FM_FTRL, FTRL

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def magic_features(col_name):
    target_features = []
    agg_features = {
        'Target'+col_name[:4]+'_mean': np.mean, 
        'Target'+col_name[:4]+'_std': np.std, 
        'Target'+col_name[:4]+'_sum': np.sum,
        }

    for fold in range(n_folds):
        tmp = log_data[(log_data['Fold'] != fold) & (log_data['Fold'].notnull())].groupby(col_name)['Target'].agg(agg_features).reset_index()
        tmp = tmp[tmp[col_name].isin(list(log_data[log_data['Fold'] == fold][col_name]))]
        tmp['Fold'] = fold
        target_features.append(tmp)

    tmp = log_data[log_data['Fold'].notnull()].groupby(col_name)['Target'].agg(agg_features).reset_index()
    tmp = tmp[tmp[col_name].isin(list(log_data[log_data['Fold'].isnull()][col_name]))]
    tmp['Fold'] = np.nan
    target_features.append(tmp)
    target_features = pd.concat(target_features, axis=0)

    return target_features

def magic2data():
    target_features = pd.DataFrame()
    cols_to_use = ['TargetCust_mean', 'TargetCust_std', 'TargetCust_sum', 'TargetProd_mean', 'TargetProd_std', 'TargetProd_sum']
    stats_cols = ['_mean', '_std', '_min', '_median', '_max', '_10per', '_25per', '_75per', '_90per']
    agg_features = {
        '_mean': np.mean, 
        '_std': np.std, 
        '_min': np.min, 
        '_max': np.max, 
        '_median': np.median, 
        '_10per': lambda x: np.percentile(x, q=10), 
        '_25per': lambda x: np.percentile(x, q=25), 
        '_75per': lambda x: np.percentile(x, q=75), 
        '_90per': lambda x: np.percentile(x, q=90),
        }

    for fold in range(n_folds):
        stats_all = log_data.loc[log_data['Fold'] == fold].groupby('FileID')[cols_to_use].agg(agg_features)
        features = pd.DataFrame()
        for col in stats_cols:
            tmp = stats_all[col]
            tmp.columns = [sub_col+col for sub_col in tmp.columns]
            tmp = tmp.reset_index()
            features = pd.concat([features, tmp], axis=1)
            
        target_features = pd.concat([target_features, features], axis=0)

    stats_all = log_data.loc[log_data['Fold'].isnull()].groupby('FileID')[cols_to_use].agg(agg_features)
    features = pd.DataFrame()
    for col in stats_cols:
        tmp = stats_all[col]
        tmp.columns = [sub_col+col for sub_col in tmp.columns]
        tmp = tmp.reset_index()
        features = pd.concat([features, tmp], axis=1)

    target_features = pd.concat([target_features, features], axis=0)

    tmp = pd.Series(target_features.iloc[:,0], name='FileID').to_frame()
    target_features.drop(['FileID'], axis=True, inplace=True)
    target_features = pd.concat([tmp, target_features], axis=1)
    target_features.fillna(-1, inplace=True)

    return target_features


if __name__ == '__main__':
    # Given some parameters
    n_folds = 10
    
    # Read log datas and file_id datas
    data_path = '/disk/Tbrain/'
    log_data = pd.read_csv(data_path+'log_data.csv')
    log_data['QueryTS'] = pd.to_datetime(log_data['QueryTS'], format='%Y-%m-%d %H:%M:%S')
    log_data['ProductID'] = log_data['ProductID'].astype(str)

    data = pd.read_csv(data_path+'data.csv')
    nrow_train = data['Fold'].notnull().sum()
    print('Load data complete.')

    # Start Feature Engineering
    CustomerID_y = magic_features('CustomerID')
    log_data = pd.merge(log_data, CustomerID_y, on=['CustomerID', 'Fold'], how='left')
    print('Group by CustomerID completed.')

    ProductID_y = magic_features('ProductID')
    log_data = pd.merge(log_data, ProductID_y, on=['ProductID', 'Fold'], how='left')
    print('Group by ProductID completed.')

    target_features = magic2data()
    data = pd.merge(data, target_features, on='FileID', how='left')
    print('Shape of features related with`Target`: {0}'.format(target_features.shape))

    data.to_csv('./magic_features.csv', index=False)
    # Drop columns only useful for magic features.
    log_data.drop(log_data.columns.tolist()[5:], axis=1, inplace=True)
    del CustomerID_y, ProductID_y, target_features
    gc.collect()


    # Sequence of FileDiff
    log_data['FileDiff'] = log_data['QueryTS'].diff().dt.total_seconds()
    log_data['FileDiffgFile'] = log_data.groupby('FileID')['QueryTS'].diff().dt.total_seconds()
    log_data['FileDiffgCust'] = log_data.groupby('CustomerID')['QueryTS'].diff().dt.total_seconds()
    log_data['FileDiffgProd'] = log_data.groupby('ProductID')['QueryTS'].diff().dt.total_seconds()

    # FileDiff
    tmp = log_data.groupby('FileID')['FileDiff'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    # FileDiffgFile
    tmp = log_data.groupby('FileID')['FileDiffgFile'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    # FileDiffgCust
    tmp = log_data.groupby('FileID')['FileDiffgCust'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    # FileDiffgProd
    tmp = log_data.groupby('FileID')['FileDiffgProd'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    print('Generate list of time delta sequence complete.')

    # Numbers of FileID
    tmp = log_data.groupby('FileID').apply(len)
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'FileCount']
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Numbers of unique CustomerID
    tmp = log_data.groupby('FileID')['CustomerID'].apply(lambda x: len(np.unique(x)))
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'CustCount']
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Numbers of unique ProductID
    tmp = log_data.groupby('FileID')['ProductID'].apply(lambda x: len(np.unique(x)))
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'ProdCount']
    data = pd.merge(data, tmp, how='left', on='FileID')
    print('Count FileID completed.')

    # Numbers of idiot countings from FileId and ProductId
    tmp = log_data.groupby('FileID')['CustomerID'].apply(lambda x: len(''.join(x)))
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'DigfCustLen']
    data = pd.merge(data, tmp, how='left', on='FileID')
    tmp = log_data.groupby('FileID')['CustomerID'].apply(lambda x: sum(c.isdigit() for c in ''.join(x))/len(''.join(x)))
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'DigfCustProp']
    data = pd.merge(data, tmp, how='left', on='FileID')
    tmp = log_data.groupby('FileID')['ProductID'].apply(lambda x: len(''.join(x)))
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'DigfProdLen']
    data = pd.merge(data, tmp, how='left', on='FileID')
    tmp = log_data.groupby('FileID')['ProductID'].apply(lambda x: sum(c.isdigit() for c in ''.join(x))/len(''.join(x)))
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'DigfProdProp']
    data = pd.merge(data, tmp, how='left', on='FileID')

    # Idiot counting
    data['NumDig'] = data['FileID'].apply(lambda x: sum(c.isdigit() for c in x))
    data['NumA'] = data['FileID'].apply(lambda x: x.count('a'))
    data['NumB'] = data['FileID'].apply(lambda x: x.count('b'))
    data['NumC'] = data['FileID'].apply(lambda x: x.count('c'))
    data['NumD'] = data['FileID'].apply(lambda x: x.count('d'))
    data['NumE'] = data['FileID'].apply(lambda x: x.count('e'))
    print('Idiot count completed.')


    # Date preprocessing
    log_data['Month'] = log_data['QueryTS'].dt.month
    log_data['Day'] = log_data['QueryTS'].dt.day
    log_data['Hour'] = log_data['QueryTS'].dt.hour
    log_data['Minute'] = log_data['QueryTS'].dt.minute
    log_data['Second'] = log_data['QueryTS'].dt.second
    log_data['WoY'] = log_data['QueryTS'].dt.weekofyear
    log_data['DoW'] = log_data['QueryTS'].dt.dayofweek
    log_data['DoY'] = log_data['QueryTS'].dt.dayofyear


    # Dealing with datetime
    cols_name = ['Month', 'Day', 'Hour', 'Minute', 'Second', 'WoY', 'DoW', 'DoY']
    # Mean
    tmp = log_data.groupby('FileID')[cols_name].agg(np.mean)
    tmp.columns = [col+'_mean' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Ten percentile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=10))
    tmp.columns = [col+'_10per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # First Quartile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=25))
    tmp.columns = [col+'_25per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Third Quartile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=75))
    tmp.columns = [col+'_75per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Ninety percentile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=90))
    tmp.columns = [col+'_90per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Median
    tmp = log_data.groupby('FileID')[cols_name].agg(np.median)
    tmp.columns = [col+'_median' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Minimum
    tmp = log_data.groupby('FileID')[cols_name].agg(np.min)
    tmp.columns = [col+'_min' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # Maximum
    tmp = log_data.groupby('FileID')[cols_name].agg(np.max)
    tmp.columns = [col+'_max' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    # standard deviation
    tmp = log_data.groupby('FileID')[cols_name].agg(np.std)
    tmp.columns = [col+'_std' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp, how='left', on='FileID')
    print('Generate datetime features completed.')

    def dt_percentile(x, p):
        index = range(len(x))
        return x.iloc[np.int((np.percentile(index, p)))]    

    latest_date = log_data['QueryTS'].iloc[-1]

    # Minimum
    tmp = log_data.groupby('FileID')['QueryTS'].agg(lambda x: latest_date - np.min(x)).dt.total_seconds()
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'Duration_min']
    data = pd.merge(data, tmp.reset_index(drop=True), how='left', on='FileID')
    # Ten percentile
    tmp = log_data.groupby('FileID')['QueryTS'].agg(lambda x: latest_date - dt_percentile(x, 10)).dt.total_seconds()
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'Duration_10per']
    data = pd.merge(data, tmp.reset_index(drop=True), how='left', on='FileID')
    # First Quantile
    tmp = log_data.groupby('FileID')['QueryTS'].agg(lambda x: latest_date - dt_percentile(x, 25)).dt.total_seconds()
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'Duration_25per']
    data = pd.merge(data, tmp.reset_index(drop=True), how='left', on='FileID')
    # Median
    tmp = log_data.groupby('FileID')['QueryTS'].agg(lambda x: latest_date - dt_percentile(x, 50)).dt.total_seconds()
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'Duration_median']
    data = pd.merge(data, tmp.reset_index(drop=True), how='left', on='FileID')
    # Third Quantile
    tmp = log_data.groupby('FileID')['QueryTS'].agg(lambda x: latest_date - dt_percentile(x, 75)).dt.total_seconds()
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'Duration_75per']
    data = pd.merge(data, tmp.reset_index(drop=True), how='left', on='FileID')
    # Ninety percentile
    tmp = log_data.groupby('FileID')['QueryTS'].agg(lambda x: latest_date - dt_percentile(x, 90)).dt.total_seconds()
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'Duration_90per']
    data = pd.merge(data, tmp.reset_index(drop=True), how='left', on='FileID')
    # Maximum
    tmp = log_data.groupby('FileID')['QueryTS'].agg(lambda x: latest_date - np.max(x)).dt.total_seconds()
    tmp = tmp.to_frame().reset_index()
    tmp.columns = ['FileID', 'Duration_max']
    data = pd.merge(data, tmp.reset_index(drop=True), how='left', on='FileID')
    # Max - Min
    data['Duration_range'] = data['Duration_min'] - data['Duration_max']

    print('Generate difference datetime features completed.')


    # Count the difference time between each file was scaned
    cols_name = ['FileDiff', 'FileDiffgFile', 'FileDiffgCust', 'FileDiffgProd']
    log_data[cols_name] = log_data[cols_name].fillna(-1)
    # Mean
    tmp = log_data.groupby('FileID')[cols_name].agg(np.mean)
    tmp.columns = [col+'_mean' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # Ten percentile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=10))
    tmp.columns = [col+'_10per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # First Quartile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=25))
    tmp.columns = [col+'_25per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # Third Quartile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=75))
    tmp.columns = [col+'_75per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # Ninety percentile
    tmp = log_data.groupby('FileID')[cols_name].agg(lambda x: np.percentile(x, q=90))
    tmp.columns = [col+'_90per' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # Median
    tmp = log_data.groupby('FileID')[cols_name].agg(np.median)
    tmp.columns = [col+'_median' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # Minimum
    tmp = log_data.groupby('FileID')[cols_name].agg(np.min)
    tmp.columns = [col+'_min' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # Maximum
    tmp = log_data.groupby('FileID')[cols_name].agg(np.max)
    tmp.columns = [col+'_max' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    # standard deviation
    tmp = log_data.groupby('FileID')[cols_name].agg(np.std)
    tmp.columns = [col+'_std' for col in tmp.columns]
    tmp.reset_index(inplace=True)
    data = pd.merge(data, tmp.reset_index(), how='left', on='FileID')
    print('Datetime groupby completed.')

    data.drop(['FileID', 'Target', 'Fold'], axis=1, inplace=True)
    data.fillna(-1, inplace=True)
    data = data.apply(lambda x: (x-np.mean(x))/(np.std(x)+np.mean(x)+1))
    X_data = csr_matrix(data)
    save_sparse_csr(data_path+'/data_{0}.npz'.format('v1'), X_data)
    print('All feature extraction completed.')
