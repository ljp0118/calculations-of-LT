import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, lagrange


#垂直距离计算公式
def distance(point, start, end):
    px, py = point
    sx, sy = start
    ex, ey = end
    return abs((ey - sy)*px - (ex - sx)*py + ex*sy - ey*sx) / np.sqrt((ey - sy)**2 + (ex - sx)**2)
def distance1(point, start, end):
    px, py = point
    sx, sy = start
    ex, ey = end
    return ((ey - sy)*px - (ex - sx)*py + ex*sy - ey*sx) / np.sqrt((ey - sy)**2 + (ex - sx)**2)

#指数加常数回归曲线模型
def exp_func(x,a,b,c):
    return a*np.exp(b*x)+c

# 定义线性模型
def linear_model(params, x):
    return params[0] * x + params[1]

# 定义残差函数
def residuals(params, x, y):
    return y - linear_model(params, x)

#定义分段拟合
def fit_segments(x, y, split_index):
    # 拟合前半段
    params1 = least_squares(residuals, [1, 0], args=(x[:split_index], y[:split_index])).x
    # 拟合后半段
    params2 = least_squares(residuals, [1, 0], args=(x[split_index:], y[split_index:])).x
    return params1, params2

# 计算两条直线的交点
def find_intersection(params1, params2):
    a1, b1 = params1
    a2, b2 = params2
    x_intersect = (b2 - b1) / (a1 - a2)
    y_intersect = a1 * x_intersect + b1
    return x_intersect, y_intersect
for i in range(183):
    people_i = pd.read_csv(f"../LT/csv/people_{i}.csv")
    if len(people_i['Speed'])>4:
        x = people_i['Speed']
        y = people_i['LA']

        #Dmax方法

        # 先用三阶多项式对图像进行拟合
        z = np.polyfit(x, y, deg=3)
        # 返回一个三阶多项式的系数数组，即得到一个三阶多项式
        p = np.poly1d(z)

        # 对x进行插值处理
        x_new = np.linspace(x.min(), x.max(), 100)
        y_new = p(x_new)

        #用Dmax方法进行乳酸阈预测
        start_point=(people_i['Speed'].iloc[0],people_i['LA'].iloc[0])
        end_point=(people_i['Speed'].iloc[-1],people_i['LA'].iloc[-1])

        line_coeffs = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
        line_poly = np.poly1d(line_coeffs) #得到方程的系数


        max_distance = 0
        for j in range(len(x_new)):
            curve_point =(x_new[j],y_new[j])
            # 找到Dmax点
            distances = distance(curve_point, start_point, end_point)
            if distances > max_distance:
                max_distance=distances
                dmax_speed=curve_point[0]
                dmax_la=curve_point[1]

        print(f"第{i}个图像的Dmax乳酸阈的运动强度是：{dmax_speed} km/h, 对应的血乳酸浓度是：{dmax_la} mmol/L")

        #ModDmax方法
        la_diff = np.diff(y)
        first_point2 = np.where(la_diff >= 0.4)

        if len(first_point2[0]) > 1:
            first_point_index2 = first_point2[0][0]
            start_point2 = (x[first_point_index2 + 1], y[first_point_index2 + 1])
        else:
            print("没有找到满足条件的索引。")
            continue

        # 用ModDmax方法进行乳酸阈预测
        end_point2 = (people_i['Speed'].iloc[-1], people_i['LA'].iloc[-1])

        line_coeffs = np.polyfit([start_point2[0], end_point2[0]], [start_point2[1], end_point2[1]], 1)
        line_poly = np.poly1d(line_coeffs)  # 得到方程的系数

        max_distance2 = 0
        for j in range(len(x_new)):
            curve_point2 = (x_new[j], y_new[j])
            # 找到ModDmax点
            distances = distance1(curve_point2, start_point2, end_point2)
            if distances > max_distance2:
                max_distance2 = distances
                ModDmax_speed = curve_point2[0]
                ModDmax_la = curve_point2[1]

        print(f"第{i}个图像的ModDmax乳酸阈的运动强度是：{round(ModDmax_speed, 4)} km/h, 对应的血乳酸浓度是：{round(ModDmax_la, 4)} mmol/L")

        #Exp-Dmax方法

        popt , pcov = curve_fit(exp_func,x_new,y_new,p0=[1,1,1],maxfev=1000)
        y_new2=exp_func(x_new, *popt)
        # 用Exp-Dmax方法进行乳酸阈预测
        max_distance3 = 0
        for j in range(len(x_new)):
            curve_point3 = (x_new[j], y_new2[j])
            # 找到Exp-Dmax点
            distances = distance(curve_point3, start_point, end_point)
            if distances > max_distance3:
                max_distance3 = distances
                exp_dmax_speed = curve_point3[0]
                exp_dmax_la = curve_point3[1]

        print(f"第{i}个图像的Exp-Dmax乳酸阈的运动强度是：{exp_dmax_speed} km/h, 对应的血乳酸浓度是：{exp_dmax_la} mmol/L")

        #log-log方法
        speed_interp = np.linspace(x.min(), x.max(), 100)
        interp_func = lagrange(x, y)
        la_interp = interp_func(speed_interp)

        log_speed_interp = np.log(speed_interp)
        log_la_interp = np.log(la_interp)

        # 找到残差平方和最小的分割点
        # 定义一个最小残差的变量存储
        min_rss = float('inf')
        best_split = None
        best_params1 = None
        best_params2 = None

        # 找到最佳的分界点，使得拟合的两条直线的残差平方和最小
        for split_index in range(10, len(log_speed_interp) - 10):
            params1, params2 = fit_segments(log_speed_interp, log_la_interp, split_index)
            rss1 = np.sum(residuals(params1, log_speed_interp[:split_index], log_la_interp[:split_index]) ** 2)
            rss2 = np.sum(residuals(params2, log_speed_interp[split_index:], log_la_interp[split_index:]) ** 2)
            total_rss = rss1 + rss2
            if total_rss < min_rss:
                min_rss = total_rss
                best_split = split_index
                best_params1 = params1
                best_params2 = params2

        x_intersect, y_intersect = find_intersection(best_params1, best_params2)
        # 输出乳酸阈值
        log_log_speed = np.exp((best_params2[1] - best_params1[1]) / (best_params1[0] - best_params2[0]))
        log_log_la = np.exp(best_params1[0] * x_intersect + best_params1[1])
        print(f"第{i}个图像的log-log乳酸阈的运动强度是：{log_log_speed} km/h, 对应的血乳酸浓度是：{log_log_la} mmol/L")

        #log-log-ModDmax方法

        start_point3 = (log_log_speed, log_log_la)
        end_point3 = (people_i['Speed'].iloc[-1], people_i['LA'].iloc[-1])

        max_distance4 = 0
        for j in range(len(x_new)):
            curve_point4 = (x_new[j], y_new[j])
            # 找到Dmax点
            distances = distance1(curve_point4, start_point3, end_point3)
            if distances > max_distance4:
                max_distance4 = distances
                log_log_ModDmax_speed = curve_point4[0]
                log_log_ModDmax_la = curve_point4[1]

        print(f"第{i}个图像的log-log-ModDmax乳酸阈的运动强度是：{log_log_ModDmax_speed} km/h, 对应的血乳酸浓度是：{log_log_ModDmax_la} mmol/L")

        #log-Exp-ModDmax方法

        max_distance5 = 0
        for j in range(len(x_new)):
            curve_point5 = (x_new[j], y_new2[j])
            distances = distance1(curve_point5, start_point3, end_point3)
            if distances > max_distance5:
                max_distance5 = distances
                log_exp_ModDmax_speed = curve_point5[0]
                log_exp_ModDmax_la = curve_point5[1]

        print(f"第{i}个图像的log_exp_ModDmax乳酸阈的运动强度是：{log_exp_ModDmax_speed} km/h, 对应的血乳酸浓度是：{log_exp_ModDmax_la} mmol/L")

        # 可视化
        plt.figure(figsize=(10, 6))
        #保证中文的正常显示
        plt.rcParams['font.family'] = 'SimHei'
        #连接两条点，得到两点的直线
        plt.plot(x, y, 'o', label='data points')
        # plt.plot(dmax_speed,dmax_la,'ro')
        plt.annotate('Dmax',
             (dmax_speed,dmax_la),
             textcoords="offset points",
             #定义箭头的格
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             xytext=(0,10),
             ha='center')
        plt.plot(x_new, y_new, '-', label='fitted curve')
        plt.scatter(dmax_speed, dmax_la, label='Dmax Point', color='red')
        plt.annotate('ModDmax',
             (ModDmax_speed,ModDmax_la),
             textcoords="offset points",
             #定义箭头的格
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             xytext=(0,20),
             ha='center')
        plt.scatter(ModDmax_speed, ModDmax_la, label='ModDmax Point', color='red')
        plt.annotate('Exp-Dmax',
             (exp_dmax_speed,exp_dmax_la),
             textcoords="offset points",
             #定义箭头的格
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             xytext=(-10,20),
             ha='center')
        plt.scatter(exp_dmax_speed, exp_dmax_la, label='Exp-Dmax Point', color='red')
        plt.annotate('log-log',
             (log_log_speed,log_log_la),
             textcoords="offset points",
             #定义箭头的格
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             xytext=(0,20),
             ha='center')
        plt.scatter(log_log_speed,log_log_la, color='red', label='log-log point')
        plt.annotate('log_log_ModDmax',
             (log_log_ModDmax_speed,log_log_ModDmax_la),
             textcoords="offset points",
             #定义箭头的格
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             xytext=(-20,30),
             ha='center')
        plt.scatter(log_log_ModDmax_speed,log_log_ModDmax_la, color='red', label='log_log_ModDmax point')
        plt.annotate('log_exp_ModDmax',
             (log_exp_ModDmax_speed,log_exp_ModDmax_la),
             textcoords="offset points",
             #定义箭头的格
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
             xytext=(10,20),
             ha='center')
        plt.scatter(log_exp_ModDmax_speed,log_exp_ModDmax_la, color='red', label='log_exp_ModDmax point')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Lactate (mmol/L)')
        plt.title('Summary of calculation methods of lactate threshold')
        plt.legend()
        plt.show()

