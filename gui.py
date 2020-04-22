import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import pickle
from PIL import Image, ImageTk
from env import Env
from dqn import DeepQNetwork
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

max_round = tk.IntVar
max_step = tk.IntVar
show = tk.IntVar
number_of_agents = tk.IntVar


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
    env = Env()
    RL = []
    for i in range(number_of_agents):
        RL.append(DeepQNetwork(i))
        RL[i].load()
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


class MyApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.chosen_agent = None
        self.Canvas = None
        self.list_frame = None
        self.picture_show_frame = None
        self.button_frame = None
        self.message_show_frame = None
        # 目标队形
        self.formation_target = np.array([[0, 0],
                                          [1, 0],
                                          [2, 0],
                                          [2, 1],
                                          [2, 2],
                                          [1, 2],
                                          [0, 2],
                                          [0, 1]]) * 100 + 20
        self.pos = self.formation_target.copy()
        self.pre_pos = self.pos.copy()
        # 初始速度为0，初始加速度也为0
        self.velocity, self.accelerate = np.zeros([8, 2]), np.zeros([8, 2])
        # 初始化能量函数
        self.energy, self.done = self.calculate_energy(), None
        self.color = ['red', 'yellow', 'blue', 'green', 'black', 'pink', 'aqua', 'purple']
        self.setupUI()

        self.refresh_data()
        self.mainloop()

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
        action_list = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]]) * 10
        if order == 1:
            # 一阶模型
            self.velocity = action_list[action]
            if 0 < self.pos.any() < 1000:
                self.pos = self.pos + self.velocity
            else:
                self.done = True
        else:
            # 二阶模型
            pass

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

    def setupUI(self):
        WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
        SCREEN_WIDTH, SCREEN_HEIGHT = self.maxsize()
        self.title("编队控制器")
        self.geometry("%dx%d+%d+%d" % (WINDOW_WIDTH, WINDOW_HEIGHT, 0.1 * SCREEN_WIDTH, 0.1 * SCREEN_HEIGHT))
        self.resizable(0, 0)
        # self.attributes('-topmost', 1, '-alpha', 1)
        self["background"] = 'grey'

        self.list_frame = list_frame = tk.Frame(width=640, height=540, bg='blue', relief='sunken', bd=2)
        # self.picture_show_frame = picture_show_frame = tk.Frame(width=640, height=540, bg='red', relief='sunken', bd=2)
        self.button_frame = button_frame = tk.Frame(width=640, height=180, bg='red', relief='sunken', bd=2)
        self.message_show_frame = message_show_frame = tk.Frame(width=640, height=180, bg='green', relief='sunken',
                                                                bd=2)

        list_frame.place(x=641, y=0)
        message_show_frame.place(x=0, y=541)
        button_frame.place(x=641, y=541)

        tk.Button(button_frame, width=18, height=2, text="开始/结束", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.start).grid(column=7, row=0, padx=10, pady=2)
        tk.Button(button_frame, width=18, height=2, text="更新", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=lambda: self.update_canvas(ax)).grid(column=7, row=1, padx=10, pady=2)
        tk.Button(button_frame, width=18, height=2, text="离开", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.quit_sim).grid(column=7, row=2, padx=10, pady=2)
        # up, right, down, left, stay
        action_list = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]])

        tk.Button(button_frame, width=4, height=2, text="上", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=lambda: self.dynamic(action=0)).grid(column=1, row=1, padx=10, pady=2)
        tk.Button(button_frame, width=4, height=2, text="左", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=lambda: self.dynamic(action=3)).grid(column=0, row=2, padx=10, pady=2)
        tk.Button(button_frame, width=4, height=2, text="下", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=lambda: self.dynamic(action=2)).grid(column=1, row=2, padx=10, pady=2)
        tk.Button(button_frame, width=4, height=2, text="右", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=lambda: self.dynamic(action=1)).grid(column=2, row=2, padx=10, pady=2)

        # 显示数据x,y坐标值
        tk.Label(message_show_frame, text='x:  ', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                 width=15).grid(column=0, row=1)
        tk.Label(message_show_frame, text='y:  ', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                 width=15).grid(column=0, row=2)
        tk.Label(message_show_frame, text='00.0', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                 width=15).grid(column=1, row=1)
        tk.Label(message_show_frame, text='00.0', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                 width=15).grid(column=1, row=2)
        tk.Label(message_show_frame, text='被观测机器人', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                 width=15).grid(column=0, row=0)
        # 创建一个下拉列表
        number = tk.StringVar()
        self.chosen_agent = ttk.Combobox(message_show_frame, width=12, textvariable=number, background='#336633')
        self.chosen_agent['values'] = (1, 2, 3, 4, 5, 6, 7, 8)  # 设置下拉列表的值
        self.chosen_agent.grid(column=1, row=0)  # 设置其在界面中出现的位置  column代表列   row 代表行
        self.chosen_agent.current(0)  # 设置下拉列表默认显示的值，0为 numberChosen['values'] 的下标值
        self.chosen_agent.config(state='readonly')
        self.chosen_agent.get()

        # 在tk左上角用matplotlib画图
        fig = Figure(figsize=(6.4, 5.4), dpi=100)
        self.ax = ax = fig.add_subplot(111)  # 添加子图:1行1列第1个

        # 将绘制的图形显示到tkinter:创建属于root的canvas画布,并将图f置于画布上
        self.Canvas = FigureCanvasTkAgg(fig, master=self)
        self.Canvas.draw()  # 注意show方法已经过时了,这里改用draw
        # self.Canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.Canvas.get_tk_widget().place(x=0, y=0)  # 随窗口大小调整而调整

        # matplotlib的导航工具栏显示上来(默认是不会显示它的)
        # toolbar = NavigationToolbar2Tk(self.Canvas, self)
        # toolbar.update()
        # self.Canvas._tkcanvas.pack(side=tkinter.LEFT,  # get_tk_widget()得到的就是_tkcanvas
        #                            fill=tkinter.BOTH)

        # 显示数据x,y坐标值
        # agent 1
        tk.Label(list_frame, text='agent  ', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                 relief='sunken',
                 width=15).grid(column=0, row=0, padx=10, pady=2)
        # agent 1
        tk.Label(list_frame, text='position  ', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                 relief='sunken',
                 width=15).grid(column=1, row=0, padx=10, pady=2)

        self.agx = [tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(),
                    tk.StringVar(), tk.StringVar()]
        self.agy = [tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(), tk.StringVar(),
                    tk.StringVar(), tk.StringVar()]

        for i in range(len(self.agx)):
            self.agx[i].set(str(i))
            self.agy[i].set(str(i))
            tk.Label(list_frame, text='x' + str(i) + ':  ', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                     relief='sunken',
                     width=15).grid(column=0, row=2 * i + 1, padx=10, pady=2)
            tk.Label(list_frame, text='y' + str(i) + ':  ', bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                     relief='sunken',
                     width=15).grid(column=0, row=2 * i + 2, padx=10, pady=2)
            tk.Label(list_frame, textvariable=self.agx[i], bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                     relief='sunken',
                     width=15).grid(column=1, row=2 * i + 1, padx=10, pady=2)
            tk.Label(list_frame, textvariable=self.agy[i], bg='#336633', fg='white', font=("隶书", 12, "bold"), height=1,
                     relief='sunken',
                     width=15).grid(column=1, row=2 * i + 2, padx=10, pady=2)

    def update_canvas(self, ax):
        self.update_flag = True
        ax.clear()
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.plot(x, y, 'ro')
        self.Canvas.draw()

    def refresh_data(self):
        # 需要刷新数据的操作
        # 代码...
        self.update_canvas(self.ax)
        for i in range(len(self.agx)):
            self.agx[i].set(str(self.pos[i, 0]))
            self.agy[i].set(str(self.pos[i, 1]))

        self.after(100, self.refresh_data)  # 这里的10000单位为毫秒

    def start(self):
        print("start the program")
        # run()

    def quit_sim(self):
        self.destroy()
        import sys
        sys.exit()


if __name__ == '__main__':
    Gui = MyApp()
