import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
df_la=pd.DataFrame()
#垂直距离计算公式
def distance(point, start, end):
    px, py = point
    sx, sy = start
    ex, ey = end
    return abs((ey - sy)*px - (ex - sx)*py + ex*sy - ey*sx) / np.sqrt((ey - sy)**2 + (ex - sx)**2)

#指数加常数回归曲线模型
def exp_func(x,a,b,c):
    return a*np.exp(b*x)+c

# 将csv表中的每一行的数据都提取为一个csv文件，并按照列存储为文件
dataset = pd.read_csv('pone.0309427.s001.csv')
dataset.rename(columns={dataset.columns[0]: 'ID'}, inplace=True)

for i in range(183):
    human_i = pd.read_csv(f"../LT/csv/people_{i}.csv")
    if len(human_i) > 3:
        condition1 = np.isnan(human_i['LA'])
        human_i = human_i[~condition1]
        x = human_i['Speed']
        y = human_i['LA']

        # 插值处理
        x_new = np.linspace(x.min(), x.max(), 100)
        interp_func=interp1d(x,y,kind='cubic')
        y_new=interp_func(x_new)

        popt , pcov = curve_fit(exp_func,x_new,y_new,p0=[1,1,1],maxfev=10000)
        y_new=exp_func(x_new, *popt)

        #用Exp-Dmax方法进行乳酸阈预测
        start_point=(human_i['Speed'].iloc[0],human_i['LA'].iloc[0])
        end_point=(human_i['Speed'].iloc[-1],human_i['LA'].iloc[-1])

        line_coeffs = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
        line_poly = np.poly1d(line_coeffs) #得到方程的系数

        max_distance = 0
        for j in range(len(x_new)):
            curve_point =(x_new[j],y_new[j])
            # 找到Exp-Dmax点
            distances = distance(curve_point, start_point, end_point)
            if distances > max_distance:
                max_distance=distances
                dmax_speed=curve_point[0]
                dmax_la=curve_point[1]
        #排除异常情况
        if dmax_speed==x[len(x)-1] or dmax_speed==x[0]:
            dmax_speed = 0
            dmax_la = 0
            data = {
                'Exp_Dmax_LT_speed': dmax_speed,
                'Exp_Dmax_LT': dmax_la
            }
            new_df = pd.DataFrame([data])
            df_la = pd.concat([df_la, new_df], ignore_index=True)

        else:
            print(f"第{i}个图像的Exp-Dmax乳酸阈的运动强度是：{dmax_speed} km/h, 对应的血乳酸浓度是：{dmax_la} mmol/L")

            data={
                'Exp_Dmax_LT_speed': dmax_speed,
                'Exp_Dmax_LT':dmax_la
            }
            new_df = pd.DataFrame([data])
            df_la = pd.concat([df_la,new_df],ignore_index=True)

            # 可视化
            plt.figure(figsize=(10, 6))
            #保证中文的正常显示
            plt.rcParams['font.family'] = 'SimHei'
            plt.plot(x, y, 'o', label='data points')
            #连接两条点，得到两点的直线
            plt.plot([start_point[0], end_point[0]],[start_point[1],end_point[1]],'-',color='green')
            plt.plot(dmax_speed,dmax_la,'ro')
            plt.annotate(f'乳酸阈速度：{round(dmax_speed,4)}km/h\n乳酸浓度：{round(dmax_la,4)}/mmol/L',
                         (dmax_speed,dmax_la),
                         textcoords="offset points",
                         #定义箭头的格式
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                         # 定义文本框的格式
                         # bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black"),
                         xytext=(15,15),
                         ha='center')
            plt.plot(x_new, exp_func(x_new, *popt), '-',label='Fitted Curve')
            plt.scatter(dmax_speed, dmax_la, label='Exp-Dmax Point', color='blue')
            plt.axvline(x=dmax_speed, color='black', linestyle='--', label='LT')
            plt.xlabel('Speed (km/h)')
            plt.ylabel('Lactate (mmol/L)')
            plt.legend()
            plt.savefig(f"../LT/imgs/Exp-Dmax/Exp-Dmax_{i}.png")
            plt.show()
    else:
        dmax_speed=0
        dmax_la=0
        data={
            'Exp_Dmax_LT_speed': dmax_speed,
            'Exp_Dmax_LT':dmax_la
        }
        new_df = pd.DataFrame([data])
        df_la = pd.concat([df_la,new_df],ignore_index=True)
        continue

df_la.to_csv('../LT/csv/Exp_Dmax.csv', index=True)