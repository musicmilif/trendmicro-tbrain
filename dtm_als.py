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


def add_ngram(q, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max+1):
            for w_index in range(len(q)-n+1):
                ngrams.append(''.join(q[w_index:w_index+n]))
        return q + ngrams


def list2str(text):
    return u' '.join([str(x) for x in text])


if __name__=='__main__':
    # Read log datas and file_id datas
    data_path = './'
    log_data = pd.read_csv(data_path+'log_data.csv')
    log_data['QueryTS'] = pd.to_datetime(log_data['QueryTS'], format='%Y-%m-%d %H:%M:%S')
    log_data['ProductID'] = log_data['ProductID'].astype(str)

    data = pd.read_csv(data_path+'data.csv')
    nrow_train = data['Fold'].notnull().sum()
    print('Load data complete.')

    # Create Documents
    log_data['FileID'] = log_data['FileID'].astype('category', categories=data['FileID'].values)
    log_data['CustomerID'] = log_data['CustomerID'].astype('category')

    log_data['CustomerID_le'] = log_data['FileID'].cat.codes
    log_data['ProductID_le'] = log_data['CustomerID'].cat.codes

    # Sequence of CustomerID
    tmp = log_data.groupby('FileID')['CustomerID_le'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    # Sequence of ProductID
    tmp = log_data.groupby('FileID')['ProductID_le'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    log_data.drop(['CustomerID_le', 'ProductID_le'], axis=1, inplace=True)
    print('Sequentail lebal encoding completed.')

    wb = wordbatch.WordBatch(list2str, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0], 
                                                            "hash_size": 2 ** 29, "norm": "l2", "tf": 'binary',
                                                            "idf": None}), procs=8)
    wb.dictionary_freeze= True
    X_cust = wb.fit_transform(data['CustomerID_le'])
    X_cust = X_cust[:, np.where(X_cust.getnnz(axis=0) > 1)[0]]
    print('Shape of X_cust: {0}'.format(X_cust.shape))
    del wb
    save_sparse_csr(data_path+'/cust_{0}.npz'.format('v1'), X_cust)
    print('Vectorize `CustomerID` completed.')


    wb = wordbatch.WordBatch(list2str, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0], 
                                                            "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                            "idf": None}), procs=8)
    wb.dictionary_freeze= True
    X_prod = wb.fit_transform(data['ProductID_le'])
    X_prod = X_prod[:, np.where(X_prod.getnnz(axis=0) > 10)[0]]
    print('Shape of X_prod: {0}'.format(X_prod.shape))
    # Fit ALS
    als = implicit.als.AlternatingLeastSquares(factors=100)
    als.fit(X_prod)
    X_prod = csr_matrix(als.item_factors)
    del wb, als

    data.drop(['CustomerID_le', 'ProductID_le'], axis=1, inplace=True)
    save_sparse_csr(data_path+'/prod_{0}.npz'.format('v1'), X_prod)
    print('Vectorize `ProductID` completed.')
