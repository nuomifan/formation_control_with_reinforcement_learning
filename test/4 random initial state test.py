# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:12:23 2020

@author: demon
"""

'''
测试在
误差： mean_square_error下
优化器: optimizers.Adam(0.01)
batch_size: 1000
训练次数: 20
学习率：1
1-4随机初始位置: 0.9513 0.9465
随机初始位置: 0.9386 0.9435
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
from tensorflow.keras import layers, optimizers, models, losses
import numpy as np
from collections import deque

# 判断移动的方向
train_times = 20
learn_rate = 1
batch_size = 1000
memory_size = 100000


def move(ob):
    v = np.zeros(18)

    if ob[0] > ob[2] + 500:  # 第一个机器人在第二个机器人右边，第一个就往左移动，否则往右移动
        # action1 = 0 左
        vx = -1
    else:
        # action1 = 1 右
        vx = 1

    if ob[1] > ob[3]:  # 第一个机器人在第二个上面，第一个就往下移动，否则往上移动
        # action1 = 2 上
        vy = -1
    else:
        # action1 = 3 下
        vy = 1

    if abs(ob[0] - ob[2] - 500) > abs(ob[1] - ob[3]):  # 左右差大于上下差，那么左右移动，否则上下移动
        v[0:2] = np.array([vx, 0])
        action = 1 if vx > 0 else 0

    else:
        v[0:2] = np.array([0, vy])
        action = 3 if vy > 0 else 2

    return v, action


# 生成标准记忆库
def generate(memory_size):
    memory = deque()
    # 全随机初始位置
    # ob = np.random.randint(-1000,1000,(memory_size,18))
    # 1-4随机初始位置
    ob = np.zeros((memory_size, 18))
    ob[:, 0:4] = np.random.randint(-1000, 1000, (memory_size, 4))
    for s in ob:
        v, action = move(s)
        s_ = s + v
        reward = 1
        memory.append((s, action, reward, s_))
    return memory


memory = generate(memory_size)

model = models.Sequential()
model.add(layers.Dense(20, input_dim=18, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(10, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(10, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(4, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))

model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(0.01))
model.summary()

ob = np.array([replay[0] for replay in memory])
ob_ = np.array([replay[3] for replay in memory])

for j in range(train_times):
    print('round: ', j + 1)
    Q = model.predict(ob)
    Q_next = model.predict(ob_)
    for i, reply in enumerate(memory):
        _, action, reward, _ = reply
        Q[i][action] = Q[i][action] + learn_rate * (reward + 0.8 * np.amax(Q_next[i]) - Q[i][action])
        # Q[i][action] = reward + 0.8 * np.amax(Q_next[i])
    model.fit(ob, Q, batch_size=batch_size)

# test
accuracy = 0
test_number = 10000
# ob = np.random.randint(-1000,1000,(test_number,18))
ob = np.zeros((test_number, 18))
ob[:, 0:4] = np.random.randint(-1000, 1000, (test_number, 4))
Q = model.predict(ob)
action_eval = np.argmax(Q, 1)

for i in range(test_number):
    v, action = move(ob[i, :])
    if action == action_eval[i]:
        accuracy += 1

print(accuracy / test_number)
