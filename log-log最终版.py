import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
from scipy.interpolate import lagrange
from sklearn.linear_model import LinearRegression

df_la=pd.DataFrame()

# 定义线性模型
#params中的两个参数分别为k和b
def linear_model(params, x):
    return params[0] * x + params[1]

# 定义残差函数
#残差=数据点的y值-拟合直线上的点的y值
def residuals(params, x, y):
    return y - linear_model(params, x)

# 分段拟合
# 假定存在一个分段拟合点，然后对拟合点的左右两边进行线性拟合
def fit_segments(x, y, split_index):
    # 拟合前半段
    # .x代表返回后的优化的参数值，least_squares 函数返回的是一个 OptimizeResult 对象，有多种属性
    params1 = least_squares(residuals, [-1, 0], args=(x[:split_index], y[:split_index])).x
    # 拟合后半段
    params2 = least_squares(residuals, [1, 0], args=(x[split_index:], y[split_index:])).x
    return params1, params2

# 计算两条直线的交点
def find_intersection(params1, params2):
    #a1=params[0],b1=params[1]
    a1, b1 = params1
    a2, b2 = params2
    x_intersect = (b2 - b1) / (a1 - a2)
    y_intersect = a1 * x_intersect + b1
    return x_intersect, y_intersect

for i in range(183):
    people_i = pd.read_csv(f'../LT/csv/people_{i}.csv')
    # people_i = pd.read_csv(f'../LT/1023-1-1.csv')
    speed = people_i['Speed'].values
    lactate = people_i['LA'].values

    speed_interp = np.linspace(speed.min(), speed.max(), 100)
    interp_func = lagrange(speed, lactate)
    la_interp = interp_func(speed_interp)

    #为了画图方便
    log_speed_interp1=np.log(speed)
    log_la_interp1=np.log(lactate)

    log_speed_interp = np.log(speed_interp)
    log_la_interp = np.log(la_interp)

    # 找到残差平方和最小的分割点
    # 定义一个最小残差的变量存储
    min_rss = float('inf')
    best_split = None
    best_params1 = None
    best_params2 = None

    #找到最佳的分界点，使得拟合的两条直线的残差平方和最小
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

    x_intersect, y_intersect=find_intersection(best_params1, best_params2)

    #延长一下直线
    x_range1=np.linspace(min(log_speed_interp),x_intersect,100)
    x_range2=np.linspace(x_intersect,max(log_speed_interp),100)

    # 输出乳酸阈值
    lactate_threshold_speed = np.exp((best_params2[1]-best_params1[1])/(best_params1[0]-best_params2[0]))
    lactate_threshold = np.exp(best_params1[0]*x_intersect+best_params1[1])
    # 存储相关数据
    data = {
        'log_log_LT_speed': lactate_threshold_speed,
        'log_log_LT': lactate_threshold
    }
    new_row=pd.DataFrame([data])
    df_la = pd.concat([df_la,new_row],ignore_index=True)


    # 画图
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'DejaVu Sans'  # 或者其他支持Unicode的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    plt.rcParams['font.family'] = 'SimHei'
    plt.scatter(log_speed_interp1, log_la_interp1, label='Data Points')
    # plt.plot(speed_interp, la_interp, label='cz Curve')
    plt.plot(x_range1, linear_model(best_params1, x_range1), color='red', label='Segment 1 Fit')
    plt.plot(x_range2, linear_model(best_params2, x_range2), color='blue', label='Segment 2 Fit')
    plt.scatter([x_intersect], [y_intersect], color='green', label='Intersection point', zorder=5)
    plt.annotate(
        f'乳酸阈对应的速度：{round(lactate_threshold_speed, 4)}km/h\n乳酸浓度：{round(lactate_threshold, 4)}mmol/L',
        (x_intersect, y_intersect),
        textcoords='offset points',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        xytext=(20, 20),
        ha='center')
    plt.xlabel('Log(Speed)')
    plt.ylabel('Log(Lactate)')
    plt.title('Log-Log')
    plt.legend()
    plt.savefig(f"../LT/imgs/log-log/log-log_{i}.png")
    plt.show()
    print(f"第{i}个图像的log-log乳酸阈的运动强度是：{round(lactate_threshold_speed, 8)} km/h, 对应的血乳酸浓度是：{round(lactate_threshold, 8)} mmol/L")

df_la.to_csv('../LT/csv/log-log.csv', index=True)