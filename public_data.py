import copy
import math
from queue import Queue

import numpy as np
from FAHP import FAHPrun
from angle import *

TASK_NUM =9 # 任务数
FRIN_NUM = 3  # 边缘节点数
# TOLERENCE_TIME = 10000 # 可容忍时间

flag = False

# # # 层次优先法参数：计算复杂度5 任务的数据总量3 容忍时延1
# # TASK_Tapewide = [[1,3,5],
# #                [1/3,1,5/3],

# # 任务-边缘节点总耗时
# FRIN_TASK = FRIN_TASK_RUN + FRIN_TASK_TRANS
#
# # # 任务时序关系 [x,y] x-当前任务 y-任务下一跳

TOLERENCE_TIME = 15000
# FRIN_TASK_compute = [10, 15, 12, 11, 13]
# FRIN_TASK_BW = [100, 115, 120, 113, 115]

#  6
# EDGE= [(0, 1), (0, 2), (0, 3), (1, 5),(2,4),(4,5)]
# TASK_INIT_VALUE= [(2332, 4697, 1166.0), (2584, 4497, 1292.0), (2491, 3339, 1245.5), (2776, 3146, 1388.0), (2148, 4800, 1074.0), (2454, 3043, 1227.0)]
# FRIN_TASK_RUN= [[233, 258, 249, 277, 214, 245], [155, 172, 166, 185, 143, 163], [194, 215, 207, 231, 179, 204], [212, 234, 226, 252, 195, 223]]
# FRIN_TASK_TRANS= [[4697, 4497, 3339, 3146, 4800, 3043], [40, 39, 29, 27, 41, 26], [39, 37, 27, 26, 40, 25], [41, 39, 29, 27, 42, 26]]
# INIT_OBS= [[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]


# # # 9
EDGE = [(0,1),(0,2),(1,4),(1,6),(4,5),(4,7),(6,7),(2,3),(2,8)]
TASK_INIT_VALUE= [(2377, 4073, 1188.5), (2415, 4798, 1207.5), (2132, 3342, 1066.0), (1941, 3445, 970.5), (1346, 3214, 673.0), (2457, 4959, 1228.5), (1911, 3880, 955.5), (2394, 3529, 1197.0), (2381, 4942, 1190.5)]
FRIN_TASK_RUN= [[237, 241, 213, 194, 134, 245, 191, 239, 238], [158, 161, 142, 129, 89, 163, 127, 159, 158], [198, 201, 177, 161, 112, 204, 159, 199, 198], [216, 219, 193, 176, 122, 223, 173, 217, 216]]
FRIN_TASK_TRANS= [[4073, 4798, 3342, 3445, 3214, 4959, 3880, 3529, 4942], [35, 41, 29, 29, 27, 43, 33, 30, 42], [33, 39, 27, 28, 26, 41, 32, 29, 41], [36, 42, 29, 30, 28, 43, 34, 31, 43]]
INIT_OBS= [[1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]


# # #  13
# EDGE= [(0, 6), (0, 3), (1, 4), (1, 6), (2, 3), (2, 11), (3, 7), (3, 8), (4, 5),(4,9), (5, 10), (6, 9),(6,10), (7, 12)]
# TASK_INIT_VALUE= [(1444, 3410, 1444.0), (1335, 3454, 1335.0), (2701, 3510, 2701.0), (2147, 4864, 2147.0), (2397, 4977, 2397.0), (2236, 3211, 2236.0), (2230, 4894, 2230.0), (2698, 3609, 2698.0), (2937, 4340, 2937.0), (2333, 4170, 2333.0), (1969, 3015, 1969.0), (1477, 4039, 1477.0), (1267, 4485, 1267.0)]
# FRIN_TASK_RUN= [[144, 133, 270, 214, 239, 223, 223, 269, 293, 233, 196, 147, 126], [96, 89, 180, 143, 159, 149, 148, 179, 195, 155, 131, 98, 84], [120, 111, 225, 178, 199, 186, 185, 224, 244, 194, 164, 123, 105], [131, 121, 245, 195, 217, 203, 202, 245, 267, 212, 179, 134, 115]]
# FRIN_TASK_TRANS= [[3410, 3454, 3510, 4864, 4977, 3211, 4894, 3609, 4340, 4170, 3015, 4039, 4485], [29, 30, 30, 42, 43, 27, 42, 31, 37, 36, 26, 35, 39], [28, 28, 29, 40, 41, 26, 40, 30, 36, 34, 25, 33, 37], [30, 30, 31, 43, 44, 28, 43, 31, 38, 36, 26, 35, 39]]
# INIT_OBS= [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# 计算任务优先级并排序（用于任务345都在同一边缘节点时候优先运行，影响任务6的运行时间）
def check_priority():
    # b = np.array(TASK_Tapewide)

    # FAHP
    weight = FAHPrun(TASK_INIT_VALUE)
    c = np.array(TASK_INIT_VALUE)
    d = np.dot(c.reshape(TASK_NUM, 3), weight)

    # ## 正常AHP法
    # weight = AHP(b).cal_weight_by_arithmetic_method()
    # print(weight)
    # c = np.array(TASK_INIT_VALUE)
    # d = np.dot(c.reshape(7,3),weight)

    # ## 引入三角模糊数排序的AHP
    # weight = AHP(b).cal_weight_by_arithmetic_method().reshape(-1, 3)
    # d = anglerun(TASK_INIT_VALUE, weight)
    # # print("子任务的权重大小：\n")

    # # 评价指标模糊化
    # d=anglerun1(TASK_INIT_VALUE,TASK_Tapewide)

    return d


# 检查当前状态是否为可行解
def check_obs(obs):
    # 每个任务只能分配给一个边缘节点
    for j in range(TASK_NUM):
        count = 0
        for i in range(FRIN_NUM):
            count += obs[i][j]
            if obs[i][j] == 1:
                if TASK_INIT_VALUE[j][2] < FRIN_TASK_RUN[i][j]:
                    # print('faulse')
                    return False
        if count != 1:
            return False
    # 最坏运行时间应小于可容忍时间
    total_time = 0  # 最坏运行时间=传输时间+运行时间
    maxx = 0
    for i in range(FRIN_NUM):
        for j in range(TASK_NUM):
            if obs[i][j] == 1:
                maxx = max(maxx, FRIN_TASK_TRANS[i][j])  # 分配到节点中的最大传输时间
                total_time += FRIN_TASK_RUN[i][j]  # 累加所有运行时间
    total_time += maxx
    if total_time > TOLERENCE_TIME:
        return False
    return True


# 移动
def move(obs, op):
    _obs = copy.deepcopy(obs)
    # 寻找任务部署在哪个边缘节点
    for i in range(len(_obs)):
        if _obs[i][op] == 1:
            index = i
            break

    # 移动任务
    _obs[index][op] = 0
    _obs[(index + 1) % FRIN_NUM][op] = 1
    return _obs


def move1(obs, op):
    _obs = copy.deepcopy(obs)
    # 寻找任务部署在哪个边缘节点
    for i in range(len(_obs)):
        if _obs[i][op[0]] == 1:
            index = i
            break
    # 移动任务
    _obs[index][op[0]] = 0
    _obs[op[1]][op[0]] = 0
    return _obs


# 选择可行解
def choose_correct_obs(obs):
    _obs = copy.deepcopy(obs)
    while not check_obs(_obs):  # 初始矩阵不满足可行解时
        # 随机选取动作
        op = np.random.randint(0, TASK_NUM)  # 在-tasknum中随机选取一个
        _obs = move(_obs, op)
    return _obs


# 计算分配方案耗时  广度排序
def cal_time(obs):
    l = [([0] * TASK_NUM) for i in range(TASK_NUM)]  # 边
    r = [0 for i in range(TASK_NUM)]  # 入度
    z = {}  # 深度
    hs = {}  # 每个子任务耗时
    hh = []
    q = Queue()  # 任务队列
    hj = {}
    qq = {i: [] for i in range(TASK_NUM)}
    # print("直接前驱集合初始化",qq)
    done_list = []
    now_time = 0
    # 加边,计算入度
    for i in EDGE:
        l[i[0]][i[1]] = 1  # 边
        r[i[1]] += 1  # 任务的入度（任务命名从1开始 计算机从0开始执行）
    # print('边：' + str(l))
    # print('入度：'+str(r))
    # 将入度为0的入队
    for i in range(TASK_NUM):  # 0-6 7个任务
        if r[i] == 0:
            q.put(i)
    # 当前深度
    deep = 1
    # 当前层对应的节点数
    next_num = q.qsize()
    # 已遍历的节点数
    count = 0
    while not q.empty():  # 队列不为空
        # 已遍历的节点数等于当前层对应的节点数
        if count == next_num:
            deep += 1
            count = 0
            next_num = q.qsize()
        # 出队
        t = q.get()
        count += 1
        # 设置该节点的深度
        z[t] = deep
        # zj =[]
        # 将该节点加入相邻节点的直接前驱节点集合，相邻节点入度-1，如果入度为0，则入队
        for i in range(TASK_NUM):
            if l[t][i] == 1:
                aa = qq[i]
                aa.append(t)  # 添加对象到列表
                qq[i] = aa  # 直接前驱集合
                # for j in qq[i]:
                #     print(qq[i],"\n",qq[j])
                # zj.append(i)
                r[i] -= 1  # 入度减1
                if r[i] == 0:
                    q.put(i)
    # 每个任务对应所有前驱
    for i in range(TASK_NUM):
        for j in qq[i]:
            if qq[i] != []:
                # print(i,j)
                qq[i].extend(qq[j])
        qq[i] = set(qq[i])
        # for i in range(TASK_NUM):
        # hj[t]=zj
    # print('任务深度'+str(z))
    # # print('直接后继节点' + str(hj))
    # print('直接前驱节点'+str(qq))

    # 为边缘节点分配任务，计算任务耗时
    for i in range(FRIN_NUM):
        tt = []
        for j in range(TASK_NUM):
            if obs[i][j] == 1:  # 边缘节点分配到任务
                tt.append(j)  # 将任务序号存放到tt集合
                hs[j] = FRIN_TASK_RUN[i][j]
        hh.append(tt)
    # print('耗时'+str(hs))
    # print("任务序号"+str(hh))
    ## 任务优先级排序
    d = check_priority()
    for i in range(FRIN_NUM):
        # hh[i].sort(key=lambda x: z[x])  # 按深度排列后的任务序号  ####！！！！！！！修改 不能按照深度排序 按照直接前驱顺序排
        # print('按照深度放置任务' + str(hh))

        # # ####按照直接前驱动顺序排序
        hh[i].sort(key=lambda x: -d[x])
        if hh[i]!=[]:
            for j in range(len(hh[i])):  # 寻找第i个任务的直接前驱
                step=0
                while step<len(hh[i]):
                    for k in range(len(hh[i])): # 搜寻索引
                        # print("节点位置",i,"节点任务",hh[i],"对应任务",hh[i][j],"节点任务的所有前驱",qq[hh[i][j]])
                        if hh[i][k] in qq[hh[i][j]] and j<k:
                            # print("任务",j,"直接前驱",qq[hh[i][j]],"位置k",k,[hh[i][k]])
                            x = hh[i][k]
                            for l in range(k,j,-1):
                                hh[i][l]=hh[i][l-1]
                            hh[i][j]=x
                            # print(hh)
                        step+=1

        # print('直接前驱',qq)
        # print("按照优先级排序后", hh)
    # print("节点中的任务顺序"+str(hh))

    # 执行任务(优先级)
    k = 1
    while len(done_list) < TASK_NUM:  # 完成列表长度未达到任务数
        xx = [-1 for i in range(FRIN_NUM)]  # 结点中的任务
        temp = []  # 所有节点中正进行的任务（可并行）
        minn = math.inf

        for i in range(FRIN_NUM):
            for j in range(len(hh[i])):
                if hh[i][j] not in done_list and set(qq[hh[i][j]]).issubset(done_list) : # 判断任务是否都在序列中(优先级)
                # if hh[i][j] not in done_list and set(qq[hh[i][j]]).issubset(done_list) and z[
                #     hh[i][j]] == k:  # 判断任务是否都在序列中(深度)
                    xx[i] = hh[i][j]
                    break

        for i in range(FRIN_NUM):
            if xx[i] != -1:
                temp.append(xx[i])

        if xx == [-1 for i in range(FRIN_NUM)]:  # 无任务
            k += 1

        for i in range(len(temp)):
            minn = min(minn, hs[temp[i]])

        if minn != math.inf:
            now_time += minn
        else:
            now_time = now_time

        for i in range(len(temp)):
            hs[temp[i]] -= minn
            if hs[temp[i]] == 0:
                done_list.append(temp[i])

    # print('任务运行时间'+str(now_time))
    return now_time


def cal_factor(obs):
    sum = 0
    # 每个边缘节点至少有一个任务
    for i in range(FRIN_NUM):
        count = 0
        for j in range(TASK_NUM):
            if obs[j] == i:
                count += 1
        if count >= 1:
            sum += 1
    return sum / FRIN_NUM


def get_total_time(obs):
    # 调度时间
    total_time = cal_time(obs)
    maxx = 0
    # 加上传输时间
    for i in range(FRIN_NUM):
        for j in range(TASK_NUM):
            if obs[i][j] == 1:
                maxx = max(maxx, FRIN_TASK_TRANS[i][j])
    # print(maxx,total_time)
    total_time += maxx
    return total_time


# 获得当前奖励
def get_reward(obs):
    # 调度时间
    total_time = get_total_time(obs)
    return total_time, (1 / total_time)


# # 获得当前奖励
# def get_reward2(obs):
#     # 调度时间
#     total_time = get_total_time(obs)
#
#     return total_time, (1 / total_time)


####数据处理--取平均值
def equal(array):
    x = array.reshape(10, -1)
    xx = np.sum(x, axis=1) / 10
    print(xx)
    return xx
