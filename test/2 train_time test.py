# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:12:23 2020

@author: demon
"""


'''
测试在mean_square_error下,
优化器: optimizers.Adam(0.01)
batch_size: 10000
训练次数:
40—— 0.9534 0.893
50—— 0.8772 0.8583
'''

from tensorflow.keras import layers, optimizers, models, losses
import numpy as np
from collections import deque
# 判断移动的方向

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
def generate():
    memory = deque()
    memory_size = 100000
    for i in range(memory_size):
        ob = np.zeros(18)
        ob[0:4] = np.random.randint(-1000, 1000, 4)
        v, action = move(ob)
        ob_ = ob + v
        reward = 1
        memory.append((ob, action, reward, ob_))
    return memory


memory = generate()

model = models.Sequential()
model.add(layers.Dense(10, input_dim=18, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(10, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(10, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(4, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
# model.compile(loss=losses.mean_squared_error, optimizer=optimizers.RMSprop(0.001))
model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(0.01))
# model.compile(loss=losses.mean_squared_error, optimizer=optimizers.SGD(0.01))
model.summary()

ob = np.array([replay[0] for replay in memory])
ob_ = np.array([replay[3] for replay in memory])

train_times = 40
for j in range(train_times):
    print('round: ', j + 1)
    Q = model.predict(ob)
    Q_next = model.predict(ob_)
    for i, reply in enumerate(memory):
        _, action, reward, _ = reply
        # Q[i][action] = Q[i][action] + learn_rate * (reward + 0.8 * np.amax(Q_next[i]) - Q[i][action])
        Q[i][action] = reward + 0.8 * np.amax(Q_next[i])
    model.fit(ob, Q, batch_size=10000)

# test
accuracy = 0
test_number = 10000
ob = np.zeros((test_number, 18))
ob[:, 0:4] = np.random.randint(-1000, 1000, (test_number, 4))
Q = model.predict(ob)
action_eval = np.argmax(Q, 1)

for i in range(test_number):
    v, action = move(ob[i, :])
    if action == action_eval[i]:
        accuracy += 1

print(accuracy / test_number)

