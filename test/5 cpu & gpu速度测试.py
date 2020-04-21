# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 04:51:52 2020

@author: demon
"""

'''
测试在mean_square_error下,
优化器: optimizers.Adam(0.01)
训练次数： 20次
cpu gpu 和训练数据对训练总时长的
batch_size 
100 ——cpu—— 118.4841513633728
100 ——gpu—— 118.30212140083313

1000 ——cpu——97.42680430412292
1000 ——gpu——99.55297088623047

10000 ——cpu——96.09955143928528
10000 ——gpu——96.59414315223694

100000 ——cpu——95.63617467880249
100000 ——gpu—— 94.42226099967957
'''
# gpu ——129 147
# cpu ——86s 454.58
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import time
starttime = time.time()
from tensorflow.keras import layers, optimizers, models, losses
import numpy as np
import tensorflow as tf
from collections import deque
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 
train_times = 20
# 判断移动的方向
tf.compat.v1.disable_eager_execution()
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
model.add(layers.Dense(1000, input_dim=18, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(1000, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(1000, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
model.add(layers.Dense(4, kernel_initializer='random_uniform',
                       bias_initializer='zeros'))
# model.compile(loss=losses.mean_squared_error, optimizer=optimizers.RMSprop(0.001))
model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(0.01))
# model.compile(loss=losses.mean_squared_error, optimizer=optimizers.SGD(0.01))
model.summary()

ob = np.array([replay[0] for replay in memory])
ob_ = np.array([replay[3] for replay in memory])


for j in range(train_times):
    print('round: ', j + 1)
    Q = model.predict(ob)
    Q_next = model.predict(ob_)
    for i, reply in enumerate(memory):
        _, action, reward, _ = reply
        Q[i][action] = reward + 0.8 * np.amax(Q_next[i])
    model.fit(ob, Q, batch_size=1000)

# test
accuracy = 0
test_number = 10000
ob = np.zeros((test_number, 18))
ob[:,0:4] = np.random.randint(-1000,1000,(test_number,4))
Q = model.predict(ob)
action_eval = np.argmax(Q,1)

for i in range(test_number):
    v, action = move(ob[i,:])
    if action == action_eval[i]:
        accuracy += 1

print(accuracy / test_number)
endtime = time.time()
print(endtime - starttime)