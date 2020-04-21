# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:26:52 2020

@author: demon
"""

from tensorflow.keras import layers, optimizers, models, losses
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import tensorflow as tf
print(tf.__version__)
tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
# 判断移动的方向
train_times = 20
learn_rate = 1
batch_size = 1000
memory_size = 1000
Max_range = 200  # 机器人活动空间在0-Max_range范围内


def move(ob):
    v = np.zeros(18)
    done = False
    if ob[0] > ob[2] + 50:  # 第一个机器人在第二个机器人右边，第一个就往左移动，否则往右移动
        # action = 0 左
        vx = -1
    else:
        # action = 1 右
        vx = 1

    if ob[1] > ob[3]:  # 第一个机器人在第二个上面，第一个就往下移动，否则往上移动
        # action = 2 上
        vy = -1
    else:
        # action = 3 下
        vy = 1

    if abs(ob[0] - ob[2] - 50) + abs(ob[1] - ob[3]) < 1:
        v[0:2] = np.array([0, 0])
        action = 4
        done = True
        return v, action, done
    elif abs(ob[0] - ob[2] - 50) > abs(ob[1] - ob[3]):  # 左右差大于上下差，那么左右移动，否则上下移动
        v[0:2] = np.array([vx, 0])
        action = 1 if vx > 0 else 0
    else:
        v[0:2] = np.array([0, vy])
        action = 3 if vy > 0 else 2
    return v, action, done

# 生成非随机标准记忆库
def generate(memory_size):
    memory = deque()
    ob = np.random.randint(-100, 100, (memory_size, 18))
    j = 0
    for i in range(memory_size - 1):
        v, action, done = move(ob[i, :])
        if done:
            reward = 1
            memory.append((ob[i, :], action, reward, done, ob[i + 1]))
            x1 = ob[j:i+1, 0]
            y1 = ob[j:i+1, 1]
            x2 = ob[j+1:i-1,2]
            y2 = ob[j+1:i-1,3]
            plt.plot(x1,y1,'-')
            plt.plot(x2,y2,'o')
            j = i
            plt.pause(.01)
            pass
        else:
            ob[i + 1] = ob[i, :] + v
            reward = 1
            memory.append((ob[i, :], action, reward, done, ob[i + 1]))

    
    return memory

memory = generate(memory_size)

def build_net():
    model = models.Sequential()
    model.add(layers.Dense(20, input_dim=18, kernel_initializer='random_uniform',
                            bias_initializer='zeros'))
    model.add(layers.Dense(10, kernel_initializer='random_uniform',
                            bias_initializer='zeros'))
    model.add(layers.Dense(10, kernel_initializer='random_uniform',
                            bias_initializer='zeros'))
    model.add(layers.Dense(5, kernel_initializer='random_uniform',
                            bias_initializer='zeros'))
    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(0.01))
    model.summary()
    return model


model = build_net()
ob = np.array([replay[0] for replay in memory])
ob_ = np.array([replay[4] for replay in memory])
action_list=[]
for j in range(train_times):
    print('round: ', j + 1)
    Q = model.predict(ob)
    Q_next = model.predict(ob)
    for i, reply in enumerate(memory):
        _, action, reward,done, _ = reply
        action_list.append(action)
        if done:
            Q[i][action] = reward
        else:
            Q[i][action] = Q[i][action] + learn_rate * (reward + 0.8 * np.amax(Q_next[i]) - Q[i][action])
    model.fit(ob, Q)


