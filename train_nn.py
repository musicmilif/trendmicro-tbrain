import warnings
warnings.filterwarnings("ignore")

import datetime
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score

import tensorflow as tf
import keras.backend as K
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


# Define NN model
def nn_model():
    model = Sequential()
    model.add(Dense(4096, activation='tanh', input_dim=train_X.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))    
    model.add(Dense(128, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


if __name__ == '__main__':
    n_folds = 10
    n_bags = 30
    predsN = 0
    data_path = '/disk/Tbrain/'
    X_data = load_sparse_csr(data_path+'new_process_v1.npz')
    data = pd.read_csv(data_path+'data.csv')

    y = data['Target'].dropna()
    cv_folds = data['Fold'].dropna()
    train_set = data['FileID'].loc[data['Fold'].notnull()].values

    # Reset data
    train_X = X_data[:len(train_set)]
    test_X = X_data[len(train_set):]
    train_y = y[:len(train_set)]

    for fold in range(n_folds):
        train_idx, valid_idx = cv_folds.loc[cv_folds != fold].index, cv_folds.loc[cv_folds == fold].index

        X_train, y_train = train_X[train_idx], train_y.iloc[train_idx]
        X_valid, y_valid = train_X[valid_idx], train_y.iloc[valid_idx]
        
        for bag in range(n_bags):
            # Train model
            model = nn_model()
            model.fit(X_train, y_train, batch_size=256, epochs=10, class_weight='auto', verbose=0)
            tmpN = model.predict_proba(X_valid).squeeze()
            predsN += model.predict_proba(test_X).squeeze()
            
            print("NN validation AUC: {0:.6f}".format(roc_auc_score(y_valid, tmpN)))
            # Clearing session
            K.clear_session()
        print('='*30)

    predsN /= (n_folds*n_bags)
    print('Predict NN completed.')

    submit = pd.concat([test_set[['FileID']], pd.Series(predsN)], axis=1)
    submit.columns = ['FileID', 'Probability']
    submit.to_csv(data_path+'NN_{0}bag_{1}.csv'.format(n_bags*n_folds, re.sub('-', '', str(datetime.date.today())[5:])), index=False)
