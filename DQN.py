import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from public_data import *
from tensorboardX import SummaryWriter

#
# losss = SummaryWriter('Dqn/loss')
# rewards = SummaryWriter('Dqn/reward')

# 1.将初始矩阵按照分配矩阵分为 边缘节点的矩阵 输入神经网络
# 2.经过神经网络计算移动哪个边缘节点的任务划算
# 3.得出最佳任务分配

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 10000  # 记忆库大小
N_ACTIONS = TASK_NUM  # 输出：分配方式--变换哪个节点的任务
N_STATES = np.array(INIT_OBS).size  # 输入：三个边缘节点的分配情况


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 256)  # 3*7
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(256, N_ACTIONS)  # 7种任务
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = x.view(-1, N_STATES)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        # self.eval_net= torch.load('../75/eval_net2.pkl')
        # self.target_net = torch.load('../75/target_net2.pkl')
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # 均方损失
        self.step = 0  # 学习步长
        self.sum_space = []
        self.sum_space2 = []
        self.sum_space3 = []
        self.e_greedy = EPSILON

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # print(i_episode,x,'\n',actions_value,'\n',action)
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def update_param(self, x):
        # 缓慢增加贪婪值，前期探索性大，后期选择最佳收益
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR * x)

    def store_transition(self, s, a, r, s_):
        # s,a,r,s形变操作
        s = s.reshape(1, -1)
        s_ = s_.reshape(1, -1)
        a = np.array(a)
        a = a.reshape(-1, 1)
        r = np.array(r)
        r = r.reshape(-1, 1)

        transition = np.hstack((s, a, r, s_))  # 水平方向平铺
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 每 target_replace_iter 更新一次神经网络
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 随机选取BATCH_SIZE行
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, index=b_a)  # shape (batch, 1) 1-行 index-列  当前s的a的返回值
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate  当前s_的所有a返回值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1


dqn = DQN()
print('\nCollecting experience...')
_now_obs = choose_correct_obs(INIT_OBS)
_all_time, _reward = get_reward(_now_obs)
n = 0
for i_episode in range(100):
    btime, breward = get_reward(_now_obs)
    start = time.time()
    sumtime = 0
    s = np.array(_now_obs)
    ep_t = 0
    yep = False  # 开始计数
    while True:
        flag = True
        while flag:
            a = int(dqn.choose_action(s))
            s_ = move(s, a)
            if check_obs(s_):
                flag = False
        all_time, r = get_reward(s_)
        if r >= _reward:
            n += 1
            done = True
            _reward = r
            _all_time = all_time
        else:
            done = False
        if all_time < btime:
            btime = all_time
        dqn.store_transition(s, a, r, s_)
        # ep_t += all_time
        if dqn.memory_counter > MEMORY_CAPACITY:
            sumtime += all_time
            yep = True
            dqn.learn()
        if dqn.step == 999:
            end = time.time()
            dqn.sum_space.append(sumtime / (dqn.step + 1))
            dqn.sum_space2.append(btime)
            dqn.sum_space3.append(end - start)
            print(i_episode, sumtime / (dqn.step + 1), dqn.step, end - start, all_time)
            dqn.step = 0
            break
        if done and yep:
            end = time.time()
            dqn.sum_space.append(sumtime / (dqn.step + 1))
            dqn.sum_space2.append(all_time)
            dqn.sum_space3.append(end - start)
            print(i_episode, sumtime / (dqn.step + 1), dqn.step, end - start, all_time)
            dqn.step = 0
            break
        s = s_

# torch.save(dqn.eval_net,'../75/eval_net2.pkl')
# torch.save(dqn.target_net,'../75/target_net2.pkl')
# np.save('../75/DQN_DATA/priority.npy', dqn.sum_space[::])
# np.save('../75/DQN_DATA/priority_run.npy', dqn.sum_space2[::])
# np.save('../75/DQN_DATA/priority_time.npy', dqn.sum_space3[::])
