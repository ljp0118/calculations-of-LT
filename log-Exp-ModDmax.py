import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
from scipy.interpolate import lagrange
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
df_la=pd.DataFrame()
def distance(point, start, end):
    px, py = point
    sx, sy = start
    ex, ey = end
    return ((ey - sy)*px - (ex - sx)*py + ex*sy - ey*sx) / np.sqrt((ey - sy)**2 + (ex - sx)**2)

# 定义线性模型
def linear_model(params, x):
        return params[0] * x + params[1]

# 定义残差函数
def residuals(params, x, y):
        return y - linear_model(params, x)
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

def exp_func(x,a,b,c):
    return a*np.exp(b*x)+c

for i in range(183):
    people_i = pd.read_csv(f'../LT/csv/people_{i}.csv')
    speed = people_i['Speed'].values
    lactate = people_i['LA'].values

    speed_interp = np.linspace(speed.min(), speed.max(), 100)
    interp_func = lagrange(speed, lactate)
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
    final_x = np.exp((best_params2[1] - best_params1[1]) / (best_params1[0] - best_params2[0]))
    final_y = np.exp(best_params1[0] * x_intersect + best_params1[1])

    # 插值处理
    x = people_i['Speed']
    y = people_i['LA']

    x_new = np.linspace(x.min(), x.max(), 100)
    interp_func = lagrange(x, y)
    y_new = interp_func(x_new)

    #检测异常情况
    try:
        popt, pcov = curve_fit(exp_func, x_new, y_new, p0=[1, 1, 1], maxfev=10000)
    except Exception as e:
        data = {
            'Log_Exp_ModDmax_LT_speed': dmax_speed,
            'Log_Exp_ModDmax_LT': dmax_la
        }
        dmax_la=0
        dmax_speed=0
        new_df = pd.DataFrame([data])
        df_la = pd.concat([df_la, new_df], ignore_index=True)

        print(f'Curve fitting failed for person {i}. Error: {str(e)}')
        continue

    start_point=(final_x,final_y)
    end_point = (people_i['Speed'].iloc[-1], people_i['LA'].iloc[-1])

    line_coeffs = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
    line_poly = np.poly1d(line_coeffs)  # 得到方程的系数

    max_distance = 0
    for j in range(len(x_new)):
        curve_point =(x_new[j],y_new[j])
        # 找到Dmax点
        distances = distance(curve_point, start_point, end_point)
        if distances > max_distance:
            max_distance=distances
            dmax_speed=curve_point[0]
            dmax_la=curve_point[1]
    if dmax_speed==x[len(x)-1] or dmax_speed==x[0]:
        dmax_la=0
        dmax_speed=0
        data = {
            'Log_Exp_ModDmax_LT_speed': dmax_speed,
            'Log_Exp_ModDmax_LT': dmax_la
        }

        new_df = pd.DataFrame([data])
        df_la = pd.concat([df_la, new_df], ignore_index=True)

    else:
        data = {
            'Log_Exp_ModDmax_LT_speed': dmax_speed,
            'Log_Exp_ModDmax_LT': dmax_la
        }
        new_df = pd.DataFrame([data])
        df_la = pd.concat([df_la, new_df], ignore_index=True)

        plt.figure(figsize=(10, 6))
        plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者其他支持Unicode的字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
        plt.rcParams['font.family'] = 'SimHei'
        plt.scatter(x, y ,label='Data Points')
        plt.plot([start_point[0],end_point[0]],[start_point[1],end_point[1]],'-',color='green')
        plt.plot(x_new, exp_func(x_new, *popt), '-', label='Fitted Curve')
        plt.axvline(x=dmax_speed,color='black',linestyle='--',label='LT')
        plt.scatter(final_x,final_y, color='green', label='log-log point', zorder=5)
        plt.annotate(
            f'乳酸阈对应的速度：{round(dmax_speed, 4)}km/h\n乳酸浓度：{round(dmax_la, 4)}mmol/L',
            (dmax_speed, dmax_la),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            xytext=(20, 20),
            ha='center')
        plt.xlabel('Speed')
        plt.ylabel('Lactate')
        plt.title('log-Exp-ModDmax')
        plt.legend()
        plt.savefig(f"../LT/imgs/log-Exp-ModDmax/log-Exp-ModDmax_{i}.png")
        plt.show()
        print(f"第{i}个图像的log-Exp-ModDmax乳酸阈的运动强度是：{round(dmax_speed, 8)} km/h, 对应的血乳酸浓度是：{round(dmax_la, 8)} mmol/L")

df_la.to_csv('../LT/csv/Log_Exp_ModDmax.csv', index=True)