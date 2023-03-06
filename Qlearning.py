import numpy as np
import pandas as pd
import time
from public_data import *

pd.set_option('display.max_columns', None)


# pd.set_option('display.max_rows', None)

class QLearning(object):
    def __init__(self, actions=None, learning_rate=0.1, reward_decay=0.9, e_greedy=0.1, sum_space=None, sum_space2=None,
                 sum_space3=None, round_num=None, iter_num=None):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 初始Q表
        if sum_space is None:
            sum_space = []
        self.sum_space = sum_space
        if sum_space2 is None:
            sum_space2 = []
        self.sum_space2 = sum_space2
        if sum_space3 is None:
            sum_space3 = []
        self.sum_space3 = sum_space3

        self.round_num = round_num
        self.iter_num = iter_num

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # 添加新状态
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,  # 索引对应列
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)  # 检测该状态是否存在，不存在就新建
        # 选择最佳动作
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table.loc[observation, :]  # loc-通过行标签索引行数据
            # 最佳动作可能有多个, 随机选择
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)  # 默认为列
        else:  # epsilon的概率探索未使用的动作
            # 随机选择动作
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]  # 得到s和动作a的Q表值，即旧值
        if done == False:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 时序差分新旧更新Q表 loc-行标签索引数据

    def update_param(self):
        # 缓慢增加贪婪值，前期探索性大，后期选择最佳收益
        if self.epsilon > 0.01:
            self.epsilon *= 0.95
        else:
            pass


ql = QLearning(actions=list(range(TASK_NUM)), round_num=100, iter_num=1000)
_now_obs = choose_correct_obs(INIT_OBS)
_time, _reward = get_reward(_now_obs)
for episode in range(ql.round_num):  # 每回合得出一个平均传输时间，每回合初始不给动作，只给初始状态
    now_obs = _now_obs
    btime, breward = get_reward(now_obs)
    done = False
    start = time.time()
    sumtime = 0.
    # print("总解个数为", FRIN_NUM ** TASK_NUM, "可行解个数为", ql.q_table.shape[0],'学习率',ql.lr)
    for i in range(ql.iter_num):
        # 选择下一个动作，并且该动作转移的状态应为可行解
        now_obs = choose_correct_obs(now_obs)
        flag = True
        while flag:
            action = ql.choose_action(str(now_obs))
            next_obs = move(now_obs, action)
            # print(next_obs)
            if check_obs(next_obs):
                flag = False

        # 获取状态即时奖励
        all_time, reward = get_reward(next_obs)
        if all_time < btime:
            btime = all_time
        if reward >= _reward:
            done = True
            # ql.update_param(0.9)  # 模拟退火
            _reward = reward
            _time = all_time
        sumtime += all_time
        # 更新Q-table
        ql.learn(str(now_obs), action, reward, str(next_obs))
        if i == ql.iter_num - 1:
            ql.sum_space.append(sumtime / i + 1)
            ql.sum_space2.append(btime)
            print('**********************', episode, i, sumtime / (i + 1), next_obs, all_time, len(ql.q_table))
            break
        if done:
            ql.sum_space.append(sumtime / (i + 1))
            ql.sum_space2.append(all_time)
            print('**********************', episode, i, sumtime / (i + 1), next_obs, all_time, ql.epsilon,
                  len(ql.q_table))
            break
        # 转移当前状态
        now_obs = next_obs
    end = time.time()
    # ql.update_param()
    ql.sum_space3.append(end - start)
    print(end - start)

# np.save('../75/QL_DATA/priority.npy',ql.sum_space[::])
# np.save('../75/QL_DATA/priority_run.npy',ql.sum_space2[::])
# np.save('../75/QL_DATA/priority_time.npy',ql.sum_space3[::])
