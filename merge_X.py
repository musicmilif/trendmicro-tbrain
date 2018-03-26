import re
import gc
import datetime
import glob
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


if __name__ == '__main__':
    data_path = '/disk/Tbrain/'
    X_als = load_sparse_csr(data_path+'als_v1.npz')
    X_cust = load_sparse_csr(data_path+'cust_v1.npz')
    X_prod = load_sparse_csr(data_path+'prod_v1.npz')
    X_diff = load_sparse_csr(data_path+'diff_v1.npz')
    X_data = load_sparse_csr(data_path+'data_v1.npz')

    X_data = hstack((X_data, X_als, X_cust, X_prod, X_diff)).tocsr()
    save_sparse_csr(data_path+'new_process_{0}.npz'.format('v1'), X_data)
    print('Save preprocessed data completed.')
    del X_als, X_cust, X_prod, X_diff
    gc.collect()
