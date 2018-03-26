import warnings
warnings.filterwarnings("ignore")

import re
import gc
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

import implicit
import wordbatch
from wordbatch.extractors import WordBag, WordHash


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def list2str(text):
    return u' '.join([str(x) for x in text])


if __name__=='__main__':
    # Read log datas and file_id datas
    data_path = '/disk/Tbrain/'
    log_data = pd.read_csv(data_path+'log_data.csv')
    log_data['QueryTS'] = pd.to_datetime(log_data['QueryTS'], format='%Y-%m-%d %H:%M:%S')
    log_data['ProductID'] = log_data['ProductID'].astype(str)

    data = pd.read_csv(data_path+'data.csv')
    nrow_train = data['Fold'].notnull().sum()
    print('Load data complete.')


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

    # Get word batch
    wb = wordbatch.WordBatch(list2str, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], 
                                                            "hash_size": 2 ** 28, "norm": "l2", "tf": 'binary',
                                                            "idf": None}), procs=24)
    wb.dictionary_freeze= True
    X_diff = wb.fit_transform(data['FileDiff'])
    X_diff = X_diff[:, np.where(X_diff.getnnz(axis=0) > 1)[0]]
    print('Shape of X_cust: {0}'.format(X_diff.shape))
    del wb

    wb = wordbatch.WordBatch(list2str, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], 
                                                            "hash_size": 2 ** 28, "norm": "l2", "tf": 'binary',
                                                            "idf": None}), procs=24)
    wb.dictionary_freeze= True
    tmp = wb.fit_transform(data['FileDiffgFile'])
    tmp = tmp[:, np.where(tmp.getnnz(axis=0) > 5)[0]]
    print('Shape of X_prod: {0}'.format(tmp.shape))
    # Fit ALS
    als = implicit.als.AlternatingLeastSquares(factors=300)
    als.fit(tmp)
    tmp = csr_matrix(als.item_factors)

    del wb, als
    X_diff = hstack((X_diff, tmp)).tocsr()


    wb = wordbatch.WordBatch(list2str, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], 
                                                            "hash_size": 2 ** 28, "norm": "l2", "tf": 'binary',
                                                            "idf": None}), procs=24)
    wb.dictionary_freeze= True
    tmp = wb.fit_transform(data['FileDiffgCust'])
    tmp = tmp[:, np.where(tmp.getnnz(axis=0) > 5)[0]]
    print('Shape of X_prod: {0}'.format(tmp.shape))
    # Fit ALS
    als = implicit.als.AlternatingLeastSquares(factors=50)
    als.fit(tmp)
    tmp = csr_matrix(als.item_factors)

    del wb, als
    X_diff = hstack((X_diff, tmp)).tocsr()

    wb = wordbatch.WordBatch(list2str, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], 
                                                            "hash_size": 2 ** 28, "norm": "l2", "tf": 'binary',
                                                            "idf": None}), procs=24)
    wb.dictionary_freeze= True
    tmp = wb.fit_transform(data['FileDiffgProd'])
    tmp = tmp[:, np.where(tmp.getnnz(axis=0) > 1)[0]]
    print('Shape of X_prod: {0}'.format(tmp.shape))
    # Fit ALS
    als = implicit.als.AlternatingLeastSquares(factors=50)
    als.fit(tmp)
    tmp = csr_matrix(als.item_factors)

    del wb, als
    X_diff = hstack((X_diff, tmp)).tocsr()

    data.drop(['FileDiff', 'FileDiffgFile', 'FileDiffgCust', 'FileDiffgProd'], axis=1, inplace=True)
    del tmp
    gc.collect()
    save_sparse_csr(data_path+'/diff_{0}.npz'.format('v1'), X_diff)
    print('Vectorize `Time Difference` completed.')
