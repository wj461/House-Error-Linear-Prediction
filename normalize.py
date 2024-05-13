import numpy as np
import pandas as pd
from sklearn import preprocessing

data_o = pd.read_csv('./data/housing_data.csv')
test_o = pd.read_csv('./data/test.csv')

data = data_o.drop(columns='MEDV').values
d = preprocessing.normalize(data)
scaled_df_data = pd.DataFrame(d, columns=data_o.columns[:-1])
scaled_df_data['MEDV'] = data_o['MEDV']


test = test_o.drop(columns='MEDV')
t = preprocessing.normalize(test)
scaled_df_test = pd.DataFrame(t, columns=test_o.columns[:-1])
scaled_df_test['MEDV'] = test_o['MEDV']


#use pd to write to csv
pd.DataFrame(scaled_df_data).to_csv('./data/housing_data_normalized.csv', index=False)
pd.DataFrame(scaled_df_test).to_csv('./data/test_normalized.csv', index=False)