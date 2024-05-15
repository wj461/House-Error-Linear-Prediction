import numpy as np
import pandas as pd
from sklearn import preprocessing

# data_o = pd.read_csv('./data/housing_data.csv')
# test_o = pd.read_csv('./data/test.csv')
data_o = pd.read_csv('./data/chat_gpt.csv')
test_o = pd.read_csv('./data/chat_gpt_test.csv')

result = data_o.columns[-1]

data = data_o.drop(columns=result).values
d = preprocessing.normalize(data)
scaled_df_data = pd.DataFrame(d, columns=data_o.columns[:-1])
scaled_df_data[result] = data_o[result]


test = test_o.drop(columns=result)
t = preprocessing.normalize(test)
scaled_df_test = pd.DataFrame(t, columns=test_o.columns[:-1])
scaled_df_test[result] = test_o[result]


#use pd to write to csv
# pd.DataFrame(scaled_df_data).to_csv('./data/housing_data_normalized.csv', index=False)
# pd.DataFrame(scaled_df_test).to_csv('./data/test_normalized.csv', index=False)
pd.DataFrame(scaled_df_data).to_csv('./data/chat_gpt_normalized.csv', index=False)
pd.DataFrame(scaled_df_test).to_csv('./data/chat_gpt_test_normalized.csv', index=False)