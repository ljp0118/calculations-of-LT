import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

all_subject=pd.read_csv('../LT/csv/filter_all.csv')
# print(all_subject.iloc[0,2], all_subject.iloc[0,3])
# print(all_subject.iloc[0,1])
# for i in range(len(all_subject)):
    # 可视化
for i in range(len(all_subject)):
    plt.figure(figsize=(10, 6))
    # 保证中文的正常显示
    plt.rcParams['font.family'] = 'SimHei'

    plt.annotate('Dmax',
                 (all_subject.iloc[i,2], all_subject.iloc[i,3]),
                 textcoords="offset points",
                 # 定义箭头的格
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 xytext=(0, 10),
                 ha='center')
    plt.scatter(all_subject.iloc[i,2], all_subject.iloc[i,3], label='Dmax Point', color='red')

    plt.annotate('ModDmax',
                 (all_subject.iloc[i,8], all_subject.iloc[i,9]),
                 textcoords="offset points",
                 # 定义箭头的格
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 xytext=(0, 20),
                 ha='center')
    plt.scatter(all_subject.iloc[i,8], all_subject.iloc[i,9], label='ModDmax Point', color='yellow')

    plt.annotate('Exp-Dmax',
                 (all_subject.iloc[i,6], all_subject.iloc[i,7]),
                 textcoords="offset points",
                 # 定义箭头的格
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 xytext=(-10, 20),
                 ha='center')
    plt.scatter(all_subject.iloc[i,6], all_subject.iloc[i,7], label='Exp-Dmax Point', color='blue')

    plt.annotate('log-log',
                 (all_subject.iloc[i,4], all_subject.iloc[i,5]),
                 textcoords="offset points",
                 # 定义箭头的格
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 xytext=(0, 20),
                 ha='center')
    plt.scatter(all_subject.iloc[i,4], all_subject.iloc[i,5], color='green', label='log-log point')

    plt.annotate('log_log_ModDmax',
                 (all_subject.iloc[i,12], all_subject.iloc[i,13]),
                 textcoords="offset points",
                 # 定义箭头的格
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 xytext=(-20, 30),
                 ha='center')
    plt.scatter(all_subject.iloc[i,12], all_subject.iloc[i,13], color='pink', label='log_log_ModDmax point')

    plt.annotate('log_exp_ModDmax',
                 (all_subject.iloc[i,10], all_subject.iloc[i,11]),
                 textcoords="offset points",
                 # 定义箭头的格
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 xytext=(10, 20),
                 ha='center')
    plt.scatter(all_subject.iloc[i,10], all_subject.iloc[i,11], color='purple', label='log_exp_ModDmax point')

    plt.xlabel('Speed (km/h)')
    plt.ylabel('Lactate (mmol/L)')
    plt.title(f'{all_subject.iloc[i,1]} participant Summary of calculation methods of lactate threshold')
    plt.legend()
    plt.savefig(f'../LT/imgs/everyone/subject_{all_subject.iloc[i,1]}.png')
    plt.show()


