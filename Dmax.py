import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_la=pd.DataFrame()
#垂直距离计算公式
def distance(point, start, end):
    px, py = point
    sx, sy = start
    ex, ey = end
    return abs((ey - sy)*px - (ex - sx)*py + ex*sy - ey*sx) / np.sqrt((ey - sy)**2 + (ex - sx)**2)

# 将csv表中的每一行的数据都提取为一个csv文件，并按照列存储为文件
dataset = pd.read_csv('pone.0309427.s001.csv')
dataset.rename(columns={dataset.columns[0]: 'ID'}, inplace=True)

for i in range(len(dataset['ID'])):
    # 对csv数据值用三次多项式插值法进行插值处理
    human_i = pd.read_csv(f"../LT/csv/people_{i}.csv")
    x = human_i['Speed']
    y = human_i['LA']

    # 先用三阶多项式对图像进行拟合
    z = np.polyfit(x, y, deg=3)
    # 返回一个三阶多项式的系数数组，即得到一个三阶多项式
    p = np.poly1d(z)

    # 对x进行插值处理
    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = p(x_new)

    #用Dmax方法进行乳酸阈预测
    start_point=(human_i['Speed'].iloc[0],human_i['LA'].iloc[0])
    end_point=(human_i['Speed'].iloc[-1],human_i['LA'].iloc[-1])

    line_coeffs = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
    line_poly = np.poly1d(line_coeffs) #得到方程的系数


    max_distance = 0
    for j in range(len(x_new)):
        curve_point =(x_new[j],y_new[j])
        # 找到Dmax点
        distances = distance(curve_point, start_point, end_point)
        if distances > max_distance and curve_point!=(x[0],y[0]) and curve_point!=(x[len(x)-1],y[len(x)-1]):
            max_distance=distances
            dmax_speed=curve_point[0]
            dmax_la=curve_point[1]

    print(f"第{i}个图像的Dmax乳酸阈的运动强度是：{dmax_speed} km/h, 对应的血乳酸浓度是：{dmax_la} mmol/L")

    # 存储相关数据
    data = {
        'Dmax_LT_speed': dmax_speed,
        'Dmax_LT': dmax_la
    }
    new_row=pd.DataFrame([data])
    df_la = pd.concat([df_la,new_row],ignore_index=True)

    # 可视化
    plt.figure(figsize=(10, 6))
    #保证中文的正常显示
    plt.rcParams['font.family'] = 'SimHei'
    #连接两条点，得到两点的直线
    plt.plot([start_point[0], end_point[0]],[start_point[1],end_point[1]],'-',color='green')
    plt.plot(x, y, 'o', label='data points')
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
    plt.plot(x_new, y_new, '-', label='fitted curve')
    plt.scatter(dmax_speed, dmax_la, label='Dmax Point', color='blue')
    plt.axvline(x=dmax_speed, color='black', linestyle='--', label='LT')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Lactate (mmol/L)')
    plt.legend()
    # plt.savefig(f"../LT/imgs/Dmax/Dmax_{i}.png")
    plt.show()

df_la.to_csv('../LT/csv/Dmax.csv', index=True)