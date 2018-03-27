import warnings
warnings.filterwarnings("ignore")

import re
import gc
import numpy as np
import pandas as pd

import implicit
from scipy.sparse import csr_matrix, hstack


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


if __name__=='__main__':
    # Read log datas and file_id datas
    data_path = './'
    log_data = pd.read_csv(data_path+'log_data.csv')
    log_data['QueryTS'] = pd.to_datetime(log_data['QueryTS'], format='%Y-%m-%d %H:%M:%S')
    log_data['ProductID'] = log_data['ProductID'].astype(str)

    data = pd.read_csv(data_path+'data.csv')
    nrow_train = data['Fold'].notnull().sum()
    print('Load data complete.')

    # Create User-Item Matrix
    log_data['FileID'] = log_data['FileID'].astype('category', categories=data['FileID'].values)
    log_data['CustomerID'] = log_data['CustomerID'].astype('category')

    row = log_data['FileID'].cat.codes
    col = log_data['CustomerID'].cat.codes

    ui = csr_matrix((np.ones(len(log_data)), (row, col)))
    ui = ui[:, np.where(ui.getnnz(axis=0) > 1)[0]]
    print('Shape of User-Item Matrix: {0}'.format(ui.shape))

    # Use Alternating Least Square to reduce dimension (latent factor)
    als = implicit.als.AlternatingLeastSquares(factors=350)
    als.fit(ui)
    X_als = csr_matrix(als.item_factors)

    del ui
    gc.collect()
    print('Extract ALS latent factor completed.')

    save_sparse_csr(data_path+'/als_{0}.npz'.format('v1'), X_als)
