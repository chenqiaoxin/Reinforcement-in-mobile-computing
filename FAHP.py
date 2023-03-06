import numpy as np
import math
import public_data


def priority(P):
    row = np.zeros(shape=(3, 3))  # 指标的模糊矩阵
    row1 = np.zeros(shape=(3, 3))  # 指标的优先矩阵
    # print(row)
    for i in range(P.shape[0]):  # 找任务
        x = P[i]
        row[i][i] = x
    for i in range(row.shape[0]):  # 指标矩阵横坐标
        for j in range(row.shape[1]):  # 指标矩阵纵纵坐标
            row1[i][j] = compare(row[i][i], row[j][j])
    return row1


def construction(array):  # 构造模糊一致矩阵
    xx = 0
    col_sum = np.sum(array, axis=0)
    for i in range(len(array)):
        for j in range(len(array)):
            array[i][j] = (col_sum[i] - col_sum[j]) / 6 + 0.5
    col_sum = np.sum(array, axis=0)
    w = 2 / (9) * col_sum
    return w


def compare(x, y):
    if x > y:
        z = 1
    elif x == y:
        z = 0.5
    elif x < y:
        z = 0
    return z


## 熵权法修改权重
##1. 归一化目标矩阵
# 准备函数——找到矩阵每列的最值
def findMin(F):
    '''
    :param F:
    :return:  返回矩阵每列的最小值
    '''
    columnMin = []
    for column in range(F.shape[1]):
        amin = min(F[:, 0])
        bmin = min(F[:, 1])
        cmin = min(F[:, 2])
    columnMin.append([amin, bmin, cmin])
    return columnMin


def findMax(F):
    '''
    :param F:
    :return:  返回矩阵每列的最大值
    '''
    columnMax = []
    for column in range(0, F.shape[0]):
        amax = max(F[:, 0])
        bmax = max(F[:, 1])
        cmax = max(F[:, 2])
    columnMax.append([amax, bmax, cmax])
    return columnMax


# 实验参数归一化 按照隶属度函数变化
def normalization(F):
    R = []
    ## 找最小值
    # if type == 0:
    columnM = findMin(np.array(F))
    ## 找最大值
    # elif type == 1:
    columnM1 = findMax(np.array(F))
    aM = columnM[0][0]
    bM = columnM[0][1]
    cM = columnM[0][2]
    aM1 = columnM1[0][0]
    bM1 = columnM1[0][1]
    cM1 = columnM1[0][2]
    for i in range(len(F)):
        row = []
        y = ((aM1 - F[i][0]) / (aM1 - aM), (bM1 - F[i][1]) / (bM1 - bM), (cM1 - F[i][2]) / (cM1 - cM))
        row.append(y)
        R.append(row)
    return R


## 求评价指标的信息熵
def entropy(R):
    R = np.array(R).reshape(public_data.TASK_NUM, 3)
    # print(R)
    k = 1 / math.log(3)
    r = np.sum(R, axis=0)
    # print("\n",r)
    rr = R / r
    # print("\n",rr)
    rrr = np.zeros(shape=R.shape)
    for i in range(rr.shape[0]):
        for j in range(rr.shape[1]):
            if rr[i][j] != 0:
                rrr[i][j] = math.log(rr[i][j])
    R = rr * rrr
    # 信息熵
    e = -k * np.sum(R, axis=0)
    # 偏差指数
    g = (1 - e) / np.sum((1 - e), axis=0)
    return g


def revise(D, g):
    wi = (D * g) / sum(D * g)
    # print(D,g)
    # print(wi)
    return wi


def FAHPrun(array):
    # if __name__ == '__main__':
    # F=np.array(TASK_INIT_VALUE)
    p = priority(np.array([5, 3, 1]).reshape(3, 1))
    D = construction(p)  ## 因素的比重
    F = np.array(array)
    R = normalization(F)
    g = entropy(R)  # 偏差指数
    wi = revise(D, g)
    # print(wi)
    return wi
