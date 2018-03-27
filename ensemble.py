import re
import datetime
import numpy as np
import pandas as pd


if __name__ == '__main__':
    data_path = './'
    # LB: 0.9455640
    lgbm1 = pd.read_csv(data_path+'LGBM_50bag_0313.csv')
    # LB: 0.9445576
    lgbm2 = pd.read_csv(data_path+'LGBM_50bag_0314.csv', usecols=['Probability'])
    lgbm2 = lgbm2['Probability']
   # LB: 0.957024
    lgbm3 = pd.read_csv(data_path+'LGBM_50bag_0321.csv', usecols=['Probability'])
    # LB: 0.9595470
    lgbm4 = pd.read_csv(data_path+'LGBM_50bag_0322.csv', usecols=['Probability'])
    # LB: 0.9194094
    rnn1 = pd.read_csv(data_path+'RNN_30bag_0313.csv', usecols=['Probability'])
    rnn2 = pd.read_csv(data_path+'RNN_30bag_0323.csv', usecols=['Probability'])
    # LB: 0.9293533
    nn1 = pd.read_csv(data_path+'NN_50bag_0315.csv', usecols=['Probability'])
    # LB: ????
    nn2 = pd.read_csv(data_path+'NN_100bag_0322.csv', usecols=['Probability'])

    submit = lgbm1
    lgbm1 = lgbm1['Probability']

    # Linear combination
    submit['Probability'] = lgbm4*0.5 + lgbm3*0.22 + lgbm2*0.08 + rnn1*0.01 + rnn2*0.05 + nn1*0.02 + nn2*0.12
    submit.to_csv(data_path+'Ensemble_{}.csv'.format(re.sub('-', '', str(datetime.date.today())[5:])), index=False)
