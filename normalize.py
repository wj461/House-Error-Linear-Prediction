import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_o = pd.read_csv('./data/housing_data.csv')

scaler = StandardScaler()
data = scaler.fit_transform(data_o[data_o.columns[:-1]])
data = np.column_stack((data, data_o[data_o.columns[-1]]))

pd.DataFrame(data, columns=data_o.columns).to_csv('./data/housing_data_normalized.csv', index=False)


data_o = pd.read_csv('./data/chat_gpt.csv')
scaler = StandardScaler()
data = scaler.fit_transform(data_o[data_o.columns[:-1]])
data = np.column_stack((data, data_o[data_o.columns[-1]]))

pd.DataFrame(data, columns= data_o.columns).to_csv('./data/chat_gpt_normalized.csv', index=False)