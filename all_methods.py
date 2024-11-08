import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
Dmax=pd.read_csv('../LT/csv/Dmax.csv')
log_log=pd.read_csv('../LT/csv/log-log.csv')
Exp_Dmax=pd.read_csv('../LT/csv/Exp_Dmax.csv')
ModDmax=pd.read_csv('../LT/csv/ModDmax.csv')
Log_Exp_ModDmax=pd.read_csv('../LT/csv/Log_Exp_ModDmax.csv')
Log_log_ModDmax=pd.read_csv('../LT/csv/Log_log_ModDmax.csv')

df1=pd.DataFrame()
log_log=log_log.replace(0,np.nan)
log_log.dropna(inplace=True)
df1=pd.merge(Dmax,log_log,left_on='Unnamed: 0',right_on='Unnamed: 0')

df2=pd.DataFrame()
Exp_Dmax=Exp_Dmax.replace(0,np.nan)
Exp_Dmax.dropna(inplace=True)
df2=pd.merge(df1,Exp_Dmax,left_on='Unnamed: 0',right_on='Unnamed: 0',how='left')

df3=pd.DataFrame()
ModDmax=ModDmax.replace(0,np.nan)
ModDmax.dropna(inplace=True)
df3=pd.merge(df2,ModDmax,left_on='Unnamed: 0',right_on='Unnamed: 0',how='left')

df4=pd.DataFrame()
Log_Exp_ModDmax=Log_Exp_ModDmax.replace(0,np.nan)
Log_Exp_ModDmax.dropna(inplace=True)
df4=pd.merge(df3,Log_Exp_ModDmax,left_on='Unnamed: 0',right_on='Unnamed: 0',how='left')

calculation=pd.DataFrame()
Log_log_ModDmax=Log_log_ModDmax.replace(0,np.nan)
Log_log_ModDmax.dropna(inplace=True)
calculation=pd.merge(df4,Log_log_ModDmax,left_on='Unnamed: 0',right_on='Unnamed: 0',how='left')
calculation.to_csv('../LT/csv/all.csv')

Dmax_x=calculation['Dmax_LT_speed'].to_numpy()
Dmax_y=calculation['Dmax_LT'].to_numpy()

log_log_x=calculation['log_log_LT_speed'].to_numpy()
log_log_y=calculation['log_log_LT'].to_numpy()

Exp_x=calculation['Exp_Dmax_LT_speed'].to_numpy()
Exp_y=calculation['Exp_Dmax_LT'].to_numpy()

ModDmax_x=calculation['ModDmax_LT_speed'].to_numpy()
ModDmax_y=calculation['ModDmax_LT'].to_numpy()

Log_Exp_ModDmax_x=calculation['Log_Exp_ModDmax_LT_speed'].to_numpy()
Log_Exp_ModDmax_y=calculation['Log_Exp_ModDmax_LT'].to_numpy()

Log_log_ModDmax_x=calculation['Log_log_ModDmax_LT_speed'].to_numpy()
Log_log_ModDmax_y=calculation['Log_log_ModDmax_LT'].to_numpy()

calculation.dropna(inplace=True)
filter_calculation=calculation
filter_calculation.to_csv('../LT/csv/filter_all.csv')
summary_stats = filter_calculation.describe()
summary_stats.iloc[:,1:].to_csv('../LT/csv/filter_all_summary_stats.csv')
# print(summary_stats.iloc[:,1:])

#遍历整个filter_calculation进行排序
column_to_sort=['Dmax_LT','log_log_LT','Exp_Dmax_LT','ModDmax_LT','Log_Exp_ModDmax_LT','Log_log_ModDmax_LT']

number_min_Dmax = 0
number_min_log_log = 0
number_min_Exp_Dmax = 0
number_min_ModDmax = 0
number_min_Log_Exp_ModDmax = 0
number_min_Log_log_ModDmax = 0

number_second_Dmax=0
number_second_log_log=0
number_second_Exp_Dmax=0
number_second_ModDmax=0
number_second_Log_Exp_ModDmax=0
number_second_Log_log_ModDmax=0

number_third_Dmax=0
number_third_log_log=0
number_third_Exp_Dmax=0
number_third_ModDmax=0
number_third_Log_Exp_ModDmax=0
number_third_Log_log_ModDmax=0

number_forth_Dmax=0
number_forth_log_log=0
number_forth_Exp_Dmax=0
number_forth_ModDmax=0
number_forth_Log_Exp_ModDmax=0
number_forth_Log_log_ModDmax=0

number_fifth_Dmax=0
number_fifth_log_log=0
number_fifth_Exp_Dmax=0
number_fifth_ModDmax=0
number_fifth_Log_Exp_ModDmax=0
number_fifth_Log_log_ModDmax=0

number_max_Dmax = 0
number_max_log_log = 0
number_max_Exp_Dmax = 0
number_max_ModDmax = 0
number_max_Log_Exp_ModDmax = 0
number_max_Log_log_ModDmax = 0

#iterrows会遍历每一行，返回一个元组，包括行索引和行数据
for index,row in filter_calculation.iterrows():
    #sorted是对键值对进行排序，排序依据是key，items()方法使得series变成键值对类型，按照x[1]进行排序，x[0]为列名
    sort_result=sorted(row[column_to_sort].items(),key=lambda x:x[1])
    #按照键值对的形式输出
    for col,val in sort_result:
        print(f"{col}:{val}")

    if sort_result[0][0]=='Dmax_LT':
        number_min_Dmax=number_min_Dmax+1
    elif sort_result[0][0] == 'log_log_LT':
        number_min_log_log=number_min_log_log+1
    elif sort_result[0][0]=='Exp_Dmax_LT':
        number_min_Exp_Dmax=number_min_Exp_Dmax+1
    elif sort_result[0][0]=='ModDmax_LT':
        number_min_ModDmax=number_min_ModDmax+1
    elif sort_result[0][0]=='Log_Exp_ModDmax_LT':
        number_min_Log_Exp_ModDmax=number_min_Log_Exp_ModDmax+1
    else:
        number_min_Log_log_ModDmax=number_min_Log_log_ModDmax+1

    if sort_result[1][0]=='Dmax_LT':
        number_second_Dmax=number_second_Dmax+1
    elif sort_result[1][0] == 'log_log_LT':
        number_second_log_log=number_second_log_log+1
    elif sort_result[1][0]=='Exp_Dmax_LT':
        number_second_Exp_Dmax=number_second_Exp_Dmax+1
    elif sort_result[1][0]=='ModDmax_LT':
        number_second_ModDmax=number_second_ModDmax+1
    elif sort_result[1][0]=='Log_Exp_ModDmax_LT':
        number_second_Log_Exp_ModDmax=number_second_Log_Exp_ModDmax+1
    else:
        number_second_Log_log_ModDmax=number_second_Log_log_ModDmax+1

    if sort_result[2][0]=='Dmax_LT':
        number_third_Dmax=number_third_Dmax+1
    elif sort_result[2][0] == 'log_log_LT':
        number_third_log_log=number_third_log_log+1
    elif sort_result[2][0]=='Exp_Dmax_LT':
        number_third_Exp_Dmax=number_third_Exp_Dmax+1
    elif sort_result[2][0]=='ModDmax_LT':
        number_third_ModDmax=number_third_ModDmax+1
    elif sort_result[2][0]=='Log_Exp_ModDmax_LT':
        number_third_Log_Exp_ModDmax=number_third_Log_Exp_ModDmax+1
    else:
        number_third_Log_log_ModDmax=number_third_Log_log_ModDmax+1

    if sort_result[3][0]=='Dmax_LT':
        number_forth_Dmax=number_forth_Dmax+1
    elif sort_result[3][0] == 'log_log_LT':
        number_forth_log_log=number_forth_log_log+1
    elif sort_result[3][0]=='Exp_Dmax_LT':
        number_forth_Exp_Dmax=number_forth_Exp_Dmax+1
    elif sort_result[3][0]=='ModDmax_LT':
        number_forth_ModDmax=number_forth_ModDmax+1
    elif sort_result[3][0]=='Log_Exp_ModDmax_LT':
        number_forth_Log_Exp_ModDmax=number_forth_Log_Exp_ModDmax+1
    else:
        number_forth_Log_log_ModDmax=number_forth_Log_log_ModDmax+1

    if sort_result[4][0]=='Dmax_LT':
        number_fifth_Dmax=number_fifth_Dmax+1
    elif sort_result[4][0] == 'log_log_LT':
        number_fifth_log_log=number_fifth_log_log+1
    elif sort_result[4][0]=='Exp_Dmax_LT':
        number_fifth_Exp_Dmax=number_fifth_Exp_Dmax+1
    elif sort_result[4][0]=='ModDmax_LT':
        number_fifth_ModDmax=number_fifth_ModDmax+1
    elif sort_result[4][0]=='Log_Exp_ModDmax_LT':
        number_fifth_Log_Exp_ModDmax=number_fifth_Log_Exp_ModDmax+1
    else:
        number_fifth_Log_log_ModDmax=number_fifth_Log_log_ModDmax+1

    if sort_result[-1][0] == 'Dmax_LT':
        number_max_Dmax = number_max_Dmax + 1
    elif sort_result[-1][0] == 'log_log_LT':
        number_max_log_log = number_max_log_log + 1
    elif sort_result[-1][0] == 'Exp_Dmax_LT':
        number_max_Exp_Dmax = number_max_Exp_Dmax + 1
    elif sort_result[-1][0] == 'ModDmax_LT':
        number_max_ModDmax = number_max_ModDmax + 1
    elif sort_result[-1][0] == 'Log_Exp_ModDmax_LT':
        number_max_Log_Exp_ModDmax = number_max_Log_Exp_ModDmax + 1
    else:
        number_max_Log_log_ModDmax = number_max_Log_log_ModDmax + 1


categories1=['Dmax','log_log','Exp_Dmax','ModDmax','Log_Exp_ModDmax','Log_log_ModDmax']
values1=[number_min_Dmax,number_min_log_log,number_min_Exp_Dmax,number_min_ModDmax,number_min_Log_Exp_ModDmax,number_min_Log_log_ModDmax]

categories2=['Dmax','log_log','Exp_Dmax','ModDmax','Log_Exp_ModDmax','Log_log_ModDmax']
values2=[number_second_Dmax,number_second_log_log,number_second_Exp_Dmax,number_second_ModDmax,number_second_Log_Exp_ModDmax,number_second_Log_log_ModDmax]

categories3=['Dmax','log_log','Exp_Dmax','ModDmax','Log_Exp_ModDmax','Log_log_ModDmax']
values3=[number_third_Dmax,number_third_log_log,number_third_Exp_Dmax,number_third_ModDmax,number_third_Log_Exp_ModDmax,number_third_Log_log_ModDmax]

categories4=['Dmax','log_log','Exp_Dmax','ModDmax','Log_Exp_ModDmax','Log_log_ModDmax']
values4=[number_forth_Dmax,number_forth_log_log,number_forth_Exp_Dmax,number_forth_ModDmax,number_forth_Log_Exp_ModDmax,number_forth_Log_log_ModDmax]

categories5=['Dmax','log_log','Exp_Dmax','ModDmax','Log_Exp_ModDmax','Log_log_ModDmax']
values5=[number_fifth_Dmax,number_fifth_log_log,number_fifth_Exp_Dmax,number_fifth_ModDmax,number_fifth_Log_Exp_ModDmax,number_fifth_Log_log_ModDmax]

categories6=['Dmax','log_log','Exp_Dmax','ModDmax','Log_Exp_ModDmax','Log_log_ModDmax']
values6=[number_max_Dmax,number_max_log_log,number_max_Exp_Dmax,number_max_ModDmax,number_max_Log_Exp_ModDmax,number_max_Log_log_ModDmax]

plt.figure(figsize=(10,6))
#返回条形对象的列表
bars=plt.bar(categories1,values1)
for bar in bars:
    #通过get_height获得y值
    y_value = bar.get_height()
    #添加文本
    #get_x获得条形左边的x值，（x+宽度）/2等于中心的x值，y_value为条形顶部的坐标
    plt.text(bar.get_x() + bar.get_width()/2, y_value, y_value, ha='center', va='bottom')
plt.title('numbers of Min_LT')
plt.xlabel('calculation methods')
plt.ylabel('numbers of people')
plt.show()

plt.figure(figsize=(10,6))
#返回条形对象的列表
bars=plt.bar(categories2,values2)
for bar in bars:
    #通过get_height获得y值
    y_value = bar.get_height()
    #添加文本
    #get_x获得条形左边的x值，（x+宽度）/2等于中心的x值，y_value为条形顶部的坐标
    plt.text(bar.get_x() + bar.get_width()/2, y_value, y_value, ha='center', va='bottom')
plt.title('numbers of second_min_LT')
plt.xlabel('calculation methods')
plt.ylabel('numbers of people')
plt.show()

plt.figure(figsize=(10,6))
#返回条形对象的列表
bars=plt.bar(categories3,values3)
for bar in bars:
    #通过get_height获得y值
    y_value = bar.get_height()
    #添加文本
    #get_x获得条形左边的x值，（x+宽度）/2等于中心的x值，y_value为条形顶部的坐标
    plt.text(bar.get_x() + bar.get_width()/2, y_value, y_value, ha='center', va='bottom')
plt.title('numbers of firth_min_LT')
plt.xlabel('calculation methods')
plt.ylabel('numbers of people')
plt.show()

plt.figure(figsize=(10,6))
#返回条形对象的列表
bars=plt.bar(categories4,values4)
for bar in bars:
    #通过get_height获得y值
    y_value = bar.get_height()
    #添加文本
    #get_x获得条形左边的x值，（x+宽度）/2等于中心的x值，y_value为条形顶部的坐标
    plt.text(bar.get_x() + bar.get_width()/2, y_value, y_value, ha='center', va='bottom')
plt.title('numbers of forth_min_LT')
plt.xlabel('calculation methods')
plt.ylabel('numbers of people')
plt.show()

plt.figure(figsize=(10,6))
#返回条形对象的列表
bars=plt.bar(categories5,values5)
for bar in bars:
    #通过get_height获得y值
    y_value = bar.get_height()
    #添加文本
    #get_x获得条形左边的x值，（x+宽度）/2等于中心的x值，y_value为条形顶部的坐标
    plt.text(bar.get_x() + bar.get_width()/2, y_value, y_value, ha='center', va='bottom')
plt.title('numbers of fifth_min_LT')
plt.xlabel('calculation methods')
plt.ylabel('numbers of people')
plt.show()

plt.figure(figsize=(10,6))
#将柱状图的上方添加数值
bars=plt.bar(categories6,values6)
for bar in bars:
    y_value = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y_value, y_value, ha='center', va='bottom')
plt.title('numbers of Max_LT')
plt.xlabel('calculation methods')
plt.ylabel('numbers of people')
plt.show()

plt.figure(figsize=(10,6))
#显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#箱线图片
#从第二列开始，每隔1列取1列，只取乳酸阈对应的血乳酸浓度
sns.boxplot(data=calculation.iloc[:,2::2])
plt.title('Comparison of Different Calculation Methods')
plt.xlabel('Results of Different Calculation Methods',labelpad=10)
plt.xticks(fontsize=4)
#画两条直线
plt.axhline(y=3.18, color='red',linestyle='--',linewidth=2)
plt.text(0.3,3.25,'LT_mean=3.18',color='red',)
plt.ylabel('Values')
plt.show()

#热力图
plt.figure(figsize=(10,6))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
correlation_matrix = filter_calculation.iloc[:,2::2].corr()
# plt.figtext(0.28,0.915,'（奇数行对应奇数列，偶数行对应偶数列)',fontsize=12,ha='left',va='top',color='red')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            xticklabels=calculation.iloc[:,2::2].columns,
            yticklabels=calculation.iloc[:,2::2].columns)
plt.xticks(fontsize=6)
plt.yticks(fontsize=8)
plt.title('Heatmap of Correlation Between Calculation Methods',pad=20)
plt.show()

#散点图
plt.figure(figsize=(10,6))
plt.scatter(Dmax_x,Dmax_y,label='Dmax_LT')
plt.scatter(log_log_x,log_log_y,label='log_log_LT',color='red')
plt.scatter(Exp_x,Exp_y,label='Exp_Dmax_LT',color='green')
plt.scatter(ModDmax_x,ModDmax_y,label='ModDmax_LT',color='yellow')
plt.scatter(Log_Exp_ModDmax_x,Log_Exp_ModDmax_y,label='Log_Exp_ModDmax_LT',color='purple')
plt.scatter(Log_log_ModDmax_x,Log_log_ModDmax_y,label='Log_log_ModDmax_LT',color='orange')
plt.axhline(y=2,color='red',linestyle='--',linewidth=2)
plt.axhline(y=4,color='red',linestyle='--',linewidth=2)
plt.xlabel('speed')
plt.ylabel('bLa')
plt.title('LT Points of Different Calculation Methods')
plt.legend()
plt.show()

#绘制概率密度图片
# sns.kdeplot(filter_calculation.iloc[:,2], shade=True)
sns.histplot(filter_calculation.iloc[:,2])

# 添加标题和标签
plt.title('Probability Density Plot of Lactate Thresholds')
plt.xlabel('Lactate Threshold')
plt.ylabel('Density')

# 显示图形
plt.show()
