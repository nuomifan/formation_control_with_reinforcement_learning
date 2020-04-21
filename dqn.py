import numpy as np
from tensorflow.keras import layers, optimizers, models, losses
from _collections import deque
import random
import tensorflow as tf
import os

tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self, agent_id):
        self.n_actions = 5  # up right left right stay
        self.n_features = 16  # vi xi vj xj
        self.id = agent_id
        self.lr = 1  # learning rate
        self.gamma = 0.9
        self.epsilon_max = 0.9
        self.replace_target_iter = 20
        self.memory_size = 1000
        self.batch_size = 100
        self.epsilon_increment = None
        # self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max  # greedy percentage
        self.epsilon = 0.9
        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = deque(maxlen=self.memory_size)
        self.memory_counter = 0
        # consist of [target_net, evaluate_net]
        self.eval_net = self.build_net()
        self.target_net = self.build_net()

    def build_net(self):
        model = models.Sequential()
        model.add(layers.Dense(4, input_dim=self.n_features, kernel_initializer='random_uniform',
                               bias_initializer='zeros'))
        model.add(layers.Dense(self.n_actions, kernel_initializer='random_uniform',
                               bias_initializer='zeros'))
        model.compile(loss=losses.mean_squared_error, optimizer=optimizers.RMSprop(0.001))
        # model.summary()
        return model

    def store_transition(self, s, a, r, s_, done):
        # 存储信息
        self.memory.append((s, a, r, s_, done))
        self.memory_counter += 1

    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            # 根据Q表选取动作
            action = np.argmax(self.eval_net.predict(s.reshape((1, self.n_features)))[0])
        else:
            # 随机选取动作
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 当记忆库不满时不学习
        if len(self.memory) < self.memory_size:
            return

        # 记录学习的次数
        self.learn_step_counter += 1

        # 用Q估计的Q值更新Q目标的Q值
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
            print('\ntarget_params_replaced\n')

        # 从所有记忆库中采样批数据
        batch_memory = random.choices(self.memory, self.batch_size)
        s_batch = np.array([replay[0] for replay in batch_memory])
        s_next_batch = np.array([replay[3] for replay in batch_memory])

        Q = self.eval_net.predict(s_batch)
        Q_next = self.target_net.predict(s_next_batch)

        for i, replay in enumerate(batch_memory):
            _, a, reward, _, done = replay
            if done:
                Q[i][a] = reward
            else:
                # Q[i][a] = (1 - self.lr) * Q[i][a] + self.lr * (reward + self.gamma * np.amax(Q_next[i]))
                Q[i][a] = reward + self.gamma * np.amax(Q_next[i])

        self.eval_net.fit(s_batch, Q, verbose=False)
        self.epsilon += 0.001 if self.epsilon < 0.99 else 0.99
        self.learn_step_counter += 1

    def save(self):
        dirs = './saved model/'
        try:
            self.eval_net.save('dqn%d.h5' % (self.id))
        except:
            if not os.path.exists(dirs):
                os.makedirs(dirs)

    def load(self):
        try:
            self.eval_net = tf.keras.models.load_model('./saved model/dqn' + str(self.id) + '.h5')
        except:
            print("No saved model")
