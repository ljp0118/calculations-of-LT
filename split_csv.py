import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 将csv表中的每一行的数据都提取为一个csv文件，并按照列存储为文件
dataset = pd.read_csv('../LT/pone.0309427.s001.csv')
dataset.rename(columns={dataset.columns[0]: 'ID'}, inplace=True)

hr_columns = ['hr_6', 'hr_8', 'hr_10', 'hr_12', 'hr_14', 'hr_16', 'hr_18', 'hr_20', 'hr_22']
la_columns = ['la_6', 'la_8', 'la_10', 'la_12', 'la_14', 'la_16', 'la_18', 'la_20', 'la_22']

for i in range(len(dataset['ID'])):
    people_i = pd.DataFrame()
    people_i['Speed'] = [6, 8, 10, 12, 14, 16, 18, 20, 22]
    df_i = dataset[dataset['ID'] == i]
    people_i['HR'] = df_i[hr_columns].values.flatten()
    people_i['LA'] = df_i[la_columns].values.flatten()

    condition = np.isnan(people_i['LA'])
    people_i = people_i[~condition]

    people_i.to_csv(f"../LT/csv/people_{i}.csv",index=True)
