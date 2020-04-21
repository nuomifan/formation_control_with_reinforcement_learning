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
            if self.pos.any() > 10:
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
        self.attributes('-topmost', 1, '-alpha', 1)
        self["background"] = 'grey'

        self.list_frame = list_frame = tk.Frame(width=640, height=540, bg='grey', relief='sunken', bd=1)
        self.picture_show_frame = picture_show_frame = tk.Frame(width=640, height=540, bg='grey', relief='sunken', bd=1)
        self.button_frame = button_frame = tk.Frame(width=640, height=180, bg='grey', relief='sunken', bd=1)
        self.message_show_frame = message_show_frame = tk.Frame(width=640, height=180, bg='grey', relief='sunken', bd=1)

        picture_show_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NW)
        list_frame.grid(row=0, column=1, padx=10, pady=10, sticky=tk.NW)
        message_show_frame.grid(row=1, column=0, padx=10, pady=10, sticky=tk.NW)
        button_frame.grid(row=1, column=1, ipadx=30, sticky=tk.NE)

        self.Canvas = tk.Canvas(picture_show_frame, width=600, height=500, bg='white')
        self.Canvas.place(x=10, y=10)

        tk.Button(button_frame, width=18, height=2, text="开始/结束", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.start).grid(column=7, row=0)
        tk.Button(button_frame, width=18, height=2, text="离开", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.quit_sim).grid(column=7, row=1)

        tk.Button(button_frame, width=4, height=2, text="上", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.update_canvas).grid(column=1, row=0)
        tk.Button(button_frame, width=4, height=2, text="左", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.start).grid(column=0, row=0, rowspan=2)
        tk.Button(button_frame, width=4, height=2, text="下", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.start).grid(column=1, row=1)
        tk.Button(button_frame, width=4, height=2, text="右", bg='#336633', fg='white', font=("隶书", 12, "bold"),
                  relief='raised', command=self.start).grid(column=2, row=0, rowspan=2)

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

    def update_canvas(self):
        f = Figure(figsize=(6, 4), dpi=100)
        a = f.add_subplot(111)  # 添加子图:1行1列第1个

        # 生成用于绘sin图的数据
        x = np.arange(0, 3, 0.01)
        y = np.sin(2 * np.pi * x)

        # 在前面得到的子图上绘图
        a.plot(x, y)

        # 将绘制的图形显示到tkinter:创建属于root的canvas画布,并将图f置于画布上
        self.Canvas = FigureCanvasTkAgg(f, master=self.picture_show_frame)
        self.Canvas.draw()  # 注意show方法已经过时了,这里改用draw
        self.Canvas.get_tk_widget().pack(side=tkinter.TOP,  # 上对齐
                                         fill=tkinter.BOTH,  # 填充方式
                                         expand=tkinter.YES)  # 随窗口大小调整而调整


        # matplotlib的导航工具栏显示上来(默认是不会显示它的)
        toolbar = NavigationToolbar2Tk(self.Canvas, self.picture_show_frame)
        toolbar.update()
        self.Canvas._tkcanvas.pack(side=tkinter.TOP,  # get_tk_widget()得到的就是_tkcanvas
                      fill=tkinter.BOTH,
                      expand=tkinter.YES)


    def start(self):
        print("start the program")
        # run()


    def quit_sim(self):
        self.destroy()
        exit()


if __name__ == '__main__':
    Gui = MyApp()
    Gui.mainloop()
