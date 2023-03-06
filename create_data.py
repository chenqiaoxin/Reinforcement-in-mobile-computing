import random
import numpy as np
# def numsplit(n):
from public_data import TASK_NUM, FRIN_NUM

"""
    传输协议802.11p
    RSU与车辆间距离 1km
    传输速率 3-27Mbit/s  5G传输 这里设置为100-120 Mbit/s 10^6
    车辆运行速度 33m/s 120km/h
    RSU算力  10000-15000 MPOS  等于10-15GPOS
    车载算例 1-2 GPOS
    优先级因素：任务复杂度（CPU轮次）1200-3000 megacycles
              数据量大小任务数据量 3-5 MBIT 10^6 3000-5000Kbit
              容忍时延设为30-50ms
"""


def prob_value(p):  # 一个按概率连边的函数
    q = int(10 * p)
    l = [1] * q + [0] * (10 - q)
    item = random.sample(l, 1)[0]
    return item


def creatDAG(n):
    into_degree = [0] * n  # 节点入度列表
    out_degree = [0] * n  # 节点出度列表
    edges = []  # 存储边的列表
    # 拓扑序就按[1,n]的顺序，依次遍历加边
    for i in range(n - 1):
        for j in range(i + 1, n):
            if i == 0 and j == n - 1:  # 不直连入口和出口
                continue
            prob = prob_value(0.4)  # 连边的概率取0.4
            if prob:
                if out_degree[i] < 2 and into_degree[j] < 1:  # 限制节点的入度和出度不大于2
                    edges.append((i, j))  # 连边
                    into_degree[j] += 1
                    out_degree[i] += 1
    for node, id in enumerate(into_degree):  # 给所有没有入边的节点添加入口节点作父亲
        if node != 0:
            if id == 0:
                m = random.randint(0, node - 1)
                edges.append((m, node))
                out_degree[m] += 1
                into_degree[node] += 1
    # print('所有节点都有父亲的edges',edges)
    for node, od in enumerate(out_degree):  # 给所有没有出边的节点添加出口节点作儿子
        if node != n - 1:
            if od == 0:
                mm = random.random()
                if mm > 0.5:
                    m = random.randint(node + 1, n - 1)
                    edges.append((node, m))
                    out_degree[node] += 1
                    into_degree[m] += 1
    print('EDGE=', sorted(edges))


def create_TASK_INIT_VALUE(n):
    TASK_INIT_VALUE = []
    d = 0
    CAR_compute = random.randint(1, 2)  ### 运算能力 1-2 TOPS
    for i in range(n):
        a = random.randint(1200, 3000)  # 任务复杂度 CPU 120-300 megacycles
        b = random.randint(3000, 5000)  # 任务数据量 3-5 MBIT 3000-5000KBIT
        c = a / CAR_compute  # 初始容忍时延 MS
        TASK_INIT_VALUE.append((a, b, c))
        d = d + c
    print("TASK_INIT_VALUE=", TASK_INIT_VALUE)
    # ,"\nTOLERENCE_TIME=",d)
    return TASK_INIT_VALUE


def create_TASK_RUN(INIT_VALUE, n, m):  ###任务数，节点数
    TASK_RUN = []
    # x=[]
    # FRIN_TASK_compute = [10, 15, 12,11,13]
    FRIN_TASK_compute = [10, 15, 12, 11, 13]
    for i in range(m):
        a = []
        # FRIN_TASK_compute = random.randint(10, 15) ### 运算能力 10-15 TOPS
        # x.append(FRIN_TASK_compute)
        for j in range(n):
            a.append(int(INIT_VALUE[j][0] / FRIN_TASK_compute[i]))
        TASK_RUN.append((a))
    # print('FRIN_TASK_compute=',x)
    # print('FRIN_TASK_compute=', FRIN_TASK_compute)
    print('FRIN_TASK_RUN=', TASK_RUN)
    ### 单位 s


def create_TASK_TRANS(INIT_VALUE, n, m):  ###任务数，节点数
    TASK_TRANS = []
    a = []
    for i in range(m):
        x = []
        FRIN_TASK_BW = [1, 115, 120, 113, 115]
        # a.append(FRIN_TASK_BW)
        for j in range(n):
            x.append(int(INIT_VALUE[j][1] / FRIN_TASK_BW[i]))
            # a.append(random.randint(1,5))
        TASK_TRANS.append((x))
    # print("FRIN_TASK_BW=",a)
    # print("FRIN_TASK_BW=", FRIN_TASK_BW)
    print('FRIN_TASK_TRANS=', TASK_TRANS)
    ###单位 MS


def create_obs(n, m):
    INIT_OBS = []
    for i in range(m):
        a = []
        for j in range(n):
            if i == 0:
                a.append(1)
            else:
                a.append(0)
        INIT_OBS.append((a))
    print('INIT_OBS=', INIT_OBS)


if __name__ == '__main__':
    n = TASK_NUM
    m = FRIN_NUM  # 结点数
    creatDAG(n)
    TASK_INIT_VALUE = create_TASK_INIT_VALUE(n)
    # TASK_INIT_VALUE= [(2720, 4379, 2720.0), (2994, 4811, 2994.0), (2833, 3915, 2833.0), (2352, 3593, 2352.0), (2775, 3308, 2775.0), (2454, 4599, 2454.0), (1393, 4329, 1393.0)]
    # print('TASK_INIT_VALUE=',TASK_INIT_VALUE)
    create_TASK_RUN(TASK_INIT_VALUE, n, m)
    create_TASK_TRANS(TASK_INIT_VALUE, n, m)
    create_obs(n, m)
