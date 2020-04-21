import numpy as np

np.random.seed(2)

MAX_RANGE = 1000


class Env:
    def __init__(self):
        # 目标队形
        self.formation_target = np.array([[0, 0],
                                          [1, 0],
                                          [2, 0],
                                          [2, 1],
                                          [2, 2],
                                          [1, 2],
                                          [0, 2],
                                          [0, 1]]) * 200
        # 机器人当前位置,机器人前一刻的位置
        self.pos, self.pre_pos = None, None
        # 初始速度，初始加速度
        self.velocity, self.accelerate = None, None
        # 计算能量函数
        self.energy, self.done = None, None

    def reset(self):
        # 将机器人移动到随机出生位置
        # self.pos = np.random.randint(0, MAX_RANGE, (8, 2))
        self.pos = self.formation_target.copy()
        self.pre_pos = self.pos.copy()
        # 初始速度为0，初始加速度也为0
        self.velocity, self.accelerate = np.zeros([8, 2]), np.zeros([8, 2])
        # 初始化能量函数
        self.energy, self.done = self.calculate_energy(), None
        return self.pos.flatten()

    def calculate_reward(self):
        reward = np.zeros(8)
        new_energy = self.calculate_energy()
        old_energy = self.energy

        for i in range(8):
            if new_energy[i] >= old_energy[i]:
                reward[i] = -1
            else:
                reward[i] = 1
        self.energy = new_energy
        return reward

    def step(self, action):
        self.done = [False] * 8
        self.pre_pos = self.pos.copy()
        self.dynamic(action)
        reward = self.calculate_reward()
        return self.pre_pos.flatten(), self.pos.flatten(), reward, self.done

    def calculate_energy(self):
        # new Energy consider other agent action
        pos = np.append(self.pos, self.pos[0:1], axis=0)
        pre_pos = np.append(self.pre_pos, self.pre_pos[0:1], axis=0)
        formation = np.append(self.formation_target, self.formation_target[0:1], axis=0)
        # energy = np.linalg.norm(pos[:-1] - pos[1:] - (formation[:-1] - formation[1:]))
        energy = np.linalg.norm(pos[:-1] - pre_pos[1:] - (formation[:-1] - formation[1:]), axis=1)
        return energy

    def dynamic(self, action, order=1):
        # up, right, down, left, stay
        action_list = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]])
        if order == 1:
            # 一阶模型
            self.velocity = action_list[action]
            self.pos = self.pos + self.velocity
            if self.pos.any() > MAX_RANGE:
                self.done = True
        else:
            # 二阶模型
            pass
