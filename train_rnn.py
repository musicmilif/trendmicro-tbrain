import warnings
warnings.filterwarnings("ignore")

import re
import gc
import numpy as np
import pandas as pd
import datetime
from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FM_FTRL

import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, GlobalAveragePooling1D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

def get_keras_data(df):
    X = {
        'customerid': pad_sequences(df['CustomerID'], maxlen=MAX_SEQ),
        'productid': pad_sequences(df['ProductID'], maxlen=MAX_SEQ),
        'filediffgcust': pad_sequences(df['FileDiffgCust'], maxlen=MAX_SEQ),
        'TargetCust_mm': np.array(df[['TargetCust_mean_mean']]),
        'TargetCust_ms': np.array(df[['TargetCust_mean_std']]),
        'TargetCust_sm': np.array(df[['TargetCust_std_mean']]),
        'TargetCust_ss': np.array(df[['TargetCust_std_std']]),
        'file_count': np.array(df[["FileCount"]]),
        'cust_count': np.array(df[["CustCount"]]),
        'prod_count': np.array(df[["ProdCount"]]),  
        'num_dig': np.array(df[["NumDig"]]),
        'num_a': np.array(df[["NumA"]]),
        'num_b': np.array(df[["NumB"]]),
        'num_c': np.array(df[["NumC"]]),
        'num_d': np.array(df[["NumD"]]),
        'num_e': np.array(df[["NumE"]]),
    }
    return X


def rnn_model(df, lr=0.001, decay=0.0):    
    # Inputs
    customerid = Input(shape=[df['customerid'].shape[1]], name='customerid')
    productid = Input(shape=[df['productid'].shape[1]], name='productid')
    filediffgcust = Input(shape=[df['filediffgcust'].shape[1]], name='filediffgcust')
    file_count = Input(shape=[1], name='file_count')
    cust_count = Input(shape=[1], name='cust_count')
    prod_count = Input(shape=[1], name='prod_count')
    num_dig = Input(shape=[1], name='num_dig')
    num_a = Input(shape=[1], name='num_a')
    num_b = Input(shape=[1], name='num_b')
    num_c = Input(shape=[1], name='num_c')
    num_d = Input(shape=[1], name='num_d')
    num_e = Input(shape=[1], name='num_e')
    TargetCust_mm = Input(shape=[1], name='TargetCust_mm')
    TargetCust_ms = Input(shape=[1], name='TargetCust_ms')
    TargetCust_sm = Input(shape=[1], name='TargetCust_sm')
    TargetCust_ss = Input(shape=[1], name='TargetCust_ss')

    # Embeddings layers
    emb_customerid = Embedding(MAX_CUST, 80)(customerid)
    emb_productid = Embedding(MAX_PROD, 5)(productid)
    emb_filediffgcust = Embedding(MAX_TIME, 20)(filediffgcust)
    emb_file_count = Embedding(MAX_FILE, 5)(file_count)
    emb_cust_count = Embedding(MAX_FILE, 5)(cust_count)
    emb_prod_count = Embedding(MAX_FILE, 5)(prod_count)
    emb_dig_count = Embedding(MAX_COUT, 3)(num_dig)
    emb_a_count = Embedding(MAX_COUT, 3)(num_a)
    emb_b_count = Embedding(MAX_COUT, 3)(num_b)
    emb_c_count = Embedding(MAX_COUT, 3)(num_c)
    emb_d_count = Embedding(MAX_COUT, 3)(num_d)
    emb_e_count = Embedding(MAX_COUT, 3)(num_e)

    # rnn layers
    rnn_layer1 = GRU(32)(emb_customerid)
    rnn_layer2 = GRU(4)(emb_productid)
    rnn_layer3 = GRU(8)(emb_filediffgcust)
    
    # FastText
    fast_layer1 = GlobalAveragePooling1D()(emb_customerid)
    fast_layer2 = GlobalAveragePooling1D()(emb_productid)
    fast_layer3 = GlobalAveragePooling1D()(emb_filediffgcust)
    
    # main layers
    main_l = concatenate([
        TargetCust_mm,
        TargetCust_ms,
        TargetCust_sm,
        TargetCust_ss,
        Flatten()(emb_file_count),
        Flatten()(emb_cust_count),
        Flatten()(emb_prod_count),
        Flatten()(emb_dig_count),
        Flatten()(emb_a_count),
        Flatten()(emb_b_count),
        Flatten()(emb_c_count),
        Flatten()(emb_d_count),        
        Flatten()(emb_e_count),        
        fast_layer1,
        fast_layer2,
        fast_layer3,
        rnn_layer1,
        rnn_layer2,
        rnn_layer3,
    ])

    main_l = Dropout(0.3) (Dense(1024)(main_l))
    main_l = BatchNormalization()(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dropout(0.2) (Dense(32)(main_l))
    main_l = Activation('elu')(main_l)

    output = Dense(1, activation="sigmoid") (main_l)
    model = Model([customerid, productid, filediffgcust, file_count, cust_count, prod_count, 
                   TargetCust_mm, TargetCust_ms, TargetCust_sm, TargetCust_ss,
                   num_dig, num_a, num_b, num_c, num_d, num_e], output)

    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model


if __name__ == '__main__':
    dir_path = '/disk/Tbrain/'
    train_set = pd.read_csv(dir_path+'training-set.csv', header=None, names=['FileID', 'Target'])
    test_set = pd.read_csv(dir_path+'testing-set.csv', header=None, names=['FileID', 'Target'])
    train_ex = pd.read_table(dir_path+'exception/exception_train.txt', header=None, names=['FileID'])
    test_ex = pd.read_table(dir_path+'exception/exception_testing.txt', header=None, names=['FileID'])

    train_set = train_set.loc[~train_set['FileID'].isin(train_ex)]
    test_set = test_set.loc[~test_set['FileID'].isin(test_ex)]


    log_data = pd.read_csv(dir_path+'log_data.csv')
    log_data['QueryTS'] = pd.to_datetime(log_data['QueryTS'], format='%Y-%m-%d %H:%M:%S')
    log_data.sort_values(['QueryTS'], ascending=True, na_position='first', inplace=True)
    log_data.reset_index(drop=True, inplace=True)
    log_data['ProductID'] = log_data['ProductID'].astype(str)
    print('Load data complete.')

    del train_ex, test_ex
    gc.collect()

    data = pd.concat([train_set, test_set],axis=0)
    data['Target'].replace(0.5, np.nan, inplace=True)


    # Label Encoding
    log_data['CustomerID'] = log_data['CustomerID'].astype('category').cat.codes
    log_data['ProductID'] = log_data['ProductID'].astype('category').cat.codes
    # Sequence of CustomerID
    tmp = log_data.groupby('FileID')['CustomerID'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    # Sequence of ProductID
    tmp = log_data.groupby('FileID')['ProductID'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    print('Sequentail lebal encoding completed.')

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

    # Idiot counting
    data['NumDig'] = data['FileID'].apply(lambda x: sum(c.isdigit() for c in x))
    data['NumA'] = data['FileID'].apply(lambda x: x.count('a'))
    data['NumB'] = data['FileID'].apply(lambda x: x.count('b'))
    data['NumC'] = data['FileID'].apply(lambda x: x.count('c'))
    data['NumD'] = data['FileID'].apply(lambda x: x.count('d'))
    data['NumE'] = data['FileID'].apply(lambda x: x.count('e'))
    print('Idiot count completed.')

    # Dealing with datetime
    log_data['FileDiff'] = log_data['QueryTS'].diff().dt.total_seconds()
    log_data['FileDiffgCust'] = log_data.groupby('CustomerID')['QueryTS'].diff().dt.total_seconds()
    log_data.fillna(-1, inplace=True)
    # -1 will get fail from Keras
    log_data['FileDiff'] = log_data['FileDiff'].astype('category').cat.codes
    log_data['FileDiffgCust'] = log_data['FileDiffgCust'].astype('category').cat.codes

    # Sequence of FileDiff
    tmp = log_data.groupby('FileID')['FileDiff'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    # Sequence of FileDiffgCust
    tmp = log_data.groupby('FileID')['FileDiffgCust'].apply(list)
    data = pd.merge(data, tmp.to_frame().reset_index(), how='left', on='FileID')
    print('Datetime sequentail completed.')

    del tmp, log_data
    gc.collect()

    new_features = pd.read_csv('./magic_features.csv', usecols=['TargetCust_mean_mean', 'TargetCust_std_mean', \
                                                                'TargetCust_mean_std', 'TargetCust_std_std'])

    data = pd.concat([data, new_features], axis=1)


    # Set 
    MAX_SEQ = 500
    MAX_CUST = np.max(data['CustomerID'].apply(max)) + 1
    MAX_PROD = np.max(data['ProductID'].apply(max)) + 1
    MAX_TIME = int(np.max([data['FileDiff'].apply(max), 
                        data['FileDiffgCust'].apply(max)])) + 2
    MAX_FILE = int(np.max([data['FileCount'].max(), 
                        data['CustCount'].max(), 
                        data['ProdCount'].max()])) + 2
    MAX_COUT = np.max([data['NumDig'].max(), 
                    data['NumA'].max(), 
                    data['NumB'].max(),
                    data['NumC'].max(),
                    data['NumD'].max(),
                    data['NumE'].max()]) + 1


    train_X = data.loc[data['FileID'].isin(train_set['FileID'])].drop(['Target'], axis=1)
    train_y = data['Target'].loc[data['FileID'].isin(train_set['FileID'])]
    test_X = data.loc[~data['FileID'].isin(train_set['FileID'])].drop(['Target'], axis=1)
    test_X = get_keras_data(test_X)

    n_folds = 10
    n_bags = 3
    nrow_train = len(train_set)
    skf = StratifiedKFold(n_splits=n_folds, random_state=5566)
    data['Fold'] = np.nan
    for f, (_, valid_idx) in enumerate(skf.split(data['FileID'].iloc[:nrow_train], data['Target'].iloc[:nrow_train])):
        data['Fold'].iloc[valid_idx] = f

    y = data['Target']
    y = y.dropna()
    cv_folds = data['Fold'].dropna()

    train_X = data[:nrow_train]
    test_X = data[nrow_train:]
    train_y = y[:nrow_train]


    # Set hyper parameters for the model.
    BATCH_SIZE = 2**9
    epochs = 2
    predsR = 0

    for fold in range(n_folds):
        train_idx, valid_idx = cv_folds.loc[cv_folds != fold].index, cv_folds.loc[cv_folds == fold].index
        X_train, y_train = get_keras_data(train_X.iloc[train_idx]), train_y.iloc[train_idx]
        X_valid, y_valid = get_keras_data(train_X.iloc[valid_idx]), train_y.iloc[valid_idx]
        
        # Calculate learning rate decay.
        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
        nrow_train = len(y_train)
        steps = int(nrow_train / BATCH_SIZE) * epochs
        lr_init, lr_fin = 0.018, 0.0006
        lr_decay = exp_decay(lr_init, lr_fin, steps)

        for bag in range(n_bags):
            # Train model
            model = rnn_model(df=X_train, lr=lr_init, decay=lr_decay)
            model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=False)
            tmpR = model.predict(X_valid, batch_size=BATCH_SIZE)
            predsR += model.predict(test_X, batch_size=BATCH_SIZE).squeeze()
            
            print("RNN validation AUC: {0:.6f}".format(roc_auc_score(y_valid, tmpR)))
            # Clearing session
            K.clear_session()
        print('='*30)
        
    predsR /= (n_folds*n_bags)
    print('Predict RNN completed.')

    gc.collect()

    submit = pd.concat([test_set[['FileID']], pd.Series(predsR)], axis=1)
    submit.columns = ['FileID', 'Probability']
    submit.to_csv('./RNN_{0}bag_{1}.csv'.format(n_bags*n_folds, re.sub('-', '', str(datetime.date.today())[5:])),index=False)
