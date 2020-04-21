# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:21:50 2020

@author: demon
"""

'''
测试在
误差： mean_square_error下
优化器: optimizers.Adam(0.01)
batch_size: 1000
训练次数: 20
学习率：1
随机初始位置
标准化输入：0.9178 0.9334
非标准化输入：0.4455 0.3512
'''
from tensorflow.keras import layers, optimizers, models, losses
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
# 判断移动的方向
train_times = 50
learn_rate = 1
batch_size = 1000
memory_size = 10000
Max_range =2000  # 机器人活动空间在0-Max_range范围内



def move(ob):
    v = np.zeros(18)

    if ob[0] > ob[2] + 500:  # 第一个机器人在第二个机器人右边，第一个就往左移动，否则往右移动
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
    ob = np.random.randint(-1000, 1000, (memory_size, 18))
    for s in ob:
        v, action = move(s)
        s_ = s + v
        reward = 1
        memory.append((s, action, reward, s_))
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
    model.add(layers.Dense(4, kernel_initializer='random_uniform',
                           bias_initializer='zeros'))
    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(0.01))
    model.summary()
    return model


model = build_net()
ob = np.array([replay[0] for replay in memory])
ob_ = np.array([replay[3] for replay in memory])


for j in range(train_times):
    print('round: ', j + 1)
    Q = model.predict(ob)
    Q_next = model.predict(ob)
    for i, reply in enumerate(memory):
        _, action, reward, _ = reply
        Q[i][action] = Q[i][action] + learn_rate * (reward + 0.8 * np.amax(Q_next[i]) - Q[i][action])
    model.fit(ob, Q, batch_size=batch_size)

# 把输入归到0-1之间
# for j in range(train_times):
#     print('round: ', j + 1)
#     Q = model.predict((ob+ Max_range//2)/Max_range)
#     Q_next = model.predict((ob+ Max_range/2)/Max_range)
#     for i, reply in enumerate(memory):
#         _, action, reward, _ = reply
#         Q[i][action] = Q[i][action] + learn_rate * (reward + 0.8 * (np.amax(Q_next[i]) - Q[i][action]))
#     model.fit((ob+ Max_range//2)/Max_range, Q, batch_size=batch_size)

# test
accuracy = 0
test_number = 10000
ob = np.random.randint(-1000, 1000, (test_number, 18))
Q = model.predict(ob)
action_eval = np.argmax(Q, 1)

for i in range(test_number):
    v, action = move(ob[i, :])
    if action == action_eval[i]:
        accuracy += 1

print(accuracy / test_number)

# 测试程序有效性
test_size = 2000
ob_t1 = np.random.randint(-1000, 1000, (test_size, 18))
ob_t2 = ob_t1.copy()
v = np.zeros([4, 18], dtype=int)
v[0, 0] = -1
v[1, 0] = 1
v[2, 1] = -1
v[3, 1] = 1
std_action_list=[]
gen_action_list=[]
for i in range(test_size - 1):
    v1,std_a = move(ob_t1[i, :])
    action = np.argmax(model.predict(ob_t2[i:i + 1, :]))
    v2 = v[action]
    ob_t1[i + 1, :] = ob_t1[i, :] + v1
    ob_t2[i + 1, :] = ob_t2[i, :] + v2
    std_action_list.append(std_a)
    gen_action_list.append(action)

# 把输入归到0-1之间
# for i in range(test_size - 1):
#     v1, _ = move(ob_t1[i, :])
#     action = np.argmax(model.predict((ob_t2[i:i + 1, :] + Max_range//2)/Max_range))
#     v2 = v[action]
#     ob_t1[i + 1, :] = ob_t1[i, :] + v1
#     ob_t2[i + 1, :] = ob_t2[i, :] + v2


accuracy = 0
test_number = 1000
ob = np.random.randint(-1000, 1000, (test_number, 18))
action =[]

for i in range(test_number-1):
    v, a1 = move(ob[i, :])
    ob[i + 1, :] = ob[i, :] + v
    action.append(a1)
    
    
for i in range(test_number-1):
    if action[i] == action_eval[i]:
        accuracy += 1
Q = model.predict(ob)
action_eval = np.argmax(Q, 1)
print(accuracy / test_number)

acc = 0
for i in range(len(std_action_list)):
    if std_action_list[i]==gen_action_list[i]:
        acc+=1
print(acc/len(std_action_list))

x1 = ob_t1[:, 0]
y1 = ob_t1[:, 1]
x2 = ob_t1[:, 2]
y2 = ob_t1[:, 3]
x3 = ob_t2[:, 0]
y3 = ob_t2[:, 1]
plt.xlim(-200, 1500)
plt.ylim(-200, 1500)
plt.plot(x1, y1, '-')
plt.plot(x3, y3, '-')
plt.plot(x1[-1], y1[-1], 'o')
plt.plot(x3[-1], y3[-1], 'o')
plt.plot(x2, y2, 'o')
plt.legend(['1', '2', '1', '2'])
plt.title("%d" % (train_times))
plt.pause(.01)

distance = ((x1-x2)**2 + (y1 - y2)**2)**0.5
plt.plot(distance)
distance = ((x3-x2)**2 + (y3 - y2)**2)**0.5
plt.plot(distance)
distance = ((x1-x3)**2 + (y1 - y3)**2)**0.5
plt.plot(distance)
plt.legend(['1-2','2-3','1-3'])
plt.pause(.01)