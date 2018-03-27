import warnings
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


if __name__ == '__main__':
    n_folds = 10
    n_bags = 50
    predsL = 0
    data_path = './'
    X_data = load_sparse_csr(data_path+'new_process_v1.npz')
    data = pd.read_csv(data_path+'data.csv')

    y = data['Target'].dropna()
    cv_folds = data['Fold'].dropna()
    train_set = data['FileID'].loc[data['Fold'].notnull()].values


    # Set lightgbm parameters
    params = {
        'learning_rate': 0.01,
        'application': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
        'feature_fraction': 0.66,
        'max_depth': 10,
        'min_data_in_leaf': 50,
        'num_leaves': 87,
        'verbosity': -1,
        'data_random_seed': 1,
        'max_bin': 64,
        'nthread': 18
    }

    # Set data
    train_X = X_data[:len(train_set)]
    test_X = X_data[len(train_set):]
    train_y = y[:len(train_set)]

    for fold in range(n_folds):
        train_idx, valid_idx = cv_folds.loc[cv_folds != fold].index, cv_folds.loc[cv_folds == fold].index
        
        d_train = lgb.Dataset(train_X[train_idx], label=train_y[train_idx])
        d_valid = lgb.Dataset(train_X[valid_idx], label=train_y[valid_idx])
        watchlist = [d_train, d_valid]
        
        for bgs in range(n_bags):
            params['feature_fraction_seed'] = np.random.randint(10000)
            params['bagging_seed'] = np.random.randint(10000)
        
            model = lgb.train(params, train_set=d_train, num_boost_round=2600, valid_sets=watchlist, 
                            early_stopping_rounds=200, verbose_eval=False)
            
            tmpL = model.predict(train_X[valid_idx])
            tmpL = (tmpL - min(tmpL))/(max(tmpL) - min(tmpL))
            print("LGBM validation AUC: {0:.6f}".format(roc_auc_score(train_y[valid_idx], tmpL)))
            predsL += model.predict(test_X)
        print('='*30)

    predsL /= (n_folds*n_bags)
    predsL = (predsL - min(predsL))/(max(predsL) - min(predsL))
    print('Predict LGBM completed.')


    submit = pd.concat([test_set[['FileID']], pd.Series(predsL)], axis=1)
    submit.columns = ['FileID', 'Probability']
    submit.to_csv(data_path+'LGBM_{0}bag_{1}.csv'.format(n_bags*n_folds, re.sub('-', '', str(datetime.date.today())[5:])), index=False)
