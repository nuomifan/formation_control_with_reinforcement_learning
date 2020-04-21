from env import Env
from dqn import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt
from gui import MyApp
# 加速度
acc = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])
# 至多多少轮
max_round = 120
# 每一轮至多多少步
max_step = 1000
# 一共有多少机器人
number_of_agents = 8
# 几轮之后开始显示图形
show = -1


def plot(observation):
    x, y = observation[::2], observation[1::2]
    for i in range(len(x)):
        plt.text(x[i], y[i], str(i))
    plt.scatter(x, y)
    plt.xlim(-200, 1200)
    plt.ylim(-200, 1200)
    plt.pause(.001)
    plt.clf()


def run():

    step = 0
    for i in range(max_round):

        print("round: ", i + 1)

        observation = env.reset()

        for j in range(max_step):

            # 选择动作
            # 上=0，右=1，下=2，左=3，静止=4
            action = []
            # 测试第一个机器人是否能学到收敛的算法
            action[0:8] = [0, 1, 2, 3, 4, 4, 4, 4]
            # for k in range(number_of_agents):
            #     action[k] = RL[k].choose_action(observation)

            # 更新环境
            observation, observation_, reward, done = env.step(action)

            if i > show:
                plot(observation)

            # 存储记忆
            # for k in range(number_of_agents):
            #     RL[k].store_transition(observation, action[k], reward[k], observation_, done[k])

            # if (step > 100) and (step % 200 == 0):
            #     for k in range(number_of_agents):
            #         RL[k].learn()

            observation = observation_
            step = step + 1

    for i in range(number_of_agents):
        RL[i].save()
        print('model ' + str(i) + ' is saved')


if __name__ == "__main__":
    env = Env()
    RL = []
    for i in range(number_of_agents):
        RL.append(DeepQNetwork(i))
        RL[i].load()



