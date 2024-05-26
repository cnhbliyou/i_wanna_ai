import time
from ScreenViewer import ScreenViewer
import pyautogui
import numpy as np
from DQNAgent import DQNAgent
import torch
import random
from threading import Thread
import os


class iwanna:
    def __init__(self, winname):
        self.s = None
        self.s_ = None
        self.a = None
        self.a_ = None
        self.r = 0
        self.done = True
        self.sv = ScreenViewer()
        self.sv.GetHWND(winname)
        self.sv.Start()
        self.thrd = Thread(target=self.check_situation)  # 创建一个线程，目标函数是更新状态
        self.thrd.start()
        self.t_c = 0
        self.blood = 12  # 满血
        self.boss_blood = 6564  # 满血

    def reset(self):
        pyautogui.press('r')
        time.sleep(1.5)
        pyautogui.press('j')
        pyautogui.press('j')
        self.s = None
        self.s_ = None
        self.a = None
        self.a_ = None
        self.done = False
        self.blood = 12  # 满血
        self.boss_blood = 6564  # 满血

    def step(self, action):
        # 奖励设计待定, 可以设计为，造成的伤害-（已经经过的时间*一个时间比例系数）
        pyautogui.press(action, interval=0.1)
        reward = self.r
        self.r = 0
        return self.sv.i0[100:-90, 181:-221, :], reward

    def check_situation(self):
        while True:
            time.sleep(0.1)
            '''判断是否结束'''
            end_logo = pyautogui.locateOnScreen('over.jpg', confidence=0.5)
            if end_logo is not None:
                self.done = True
                continue
            '''检测boss血条'''
            boss_blood_line = self.sv.i0[68:80, 205:-250, :]
            blood_n = np.count_nonzero(boss_blood_line[boss_blood_line == [0, 0, 255]])
            change = self.boss_blood - blood_n
            if change > 0 and change < 64:
                # 造成伤害了
                self.r += change
                self.boss_blood = blood_n


if __name__ == "__main__":
    env = iwanna('雷电模拟器')
    EPISODES = 1000
    state_size = (385, 600, 3)  # 状态的维度
    action_size = 4  # 动作的数量
    agent = DQNAgent(state_size, action_size)  # 初始化DQN代理
    action_map = ['a', 'd', 'j', 'k']  # 动作映射

    # 检查是否存在已经保存的模型
    model_files = [f for f in os.listdir() if f.startswith("iwanna-dqn-") and f.endswith(".h5")]
    if model_files:
        latest_model = sorted(model_files, key=lambda x: int(x.split('-')[2].split('.')[0]))[-1]
        agent.load(latest_model)
        print(f"Loaded model {latest_model}")

    for e in range(EPISODES):
        env.reset()
        state = env.sv.i0[100:-90, 181:-221, :]
        state = np.reshape(state, [1, state_size[0], state_size[1], state_size[2]])

        while not env.done:
            time.sleep(0.1)
            action_idx = agent.act(state)  # 使用DQN代理选择动作
            action = action_map[action_idx]
            next_state, reward = env.step(action)
            next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])

            agent.remember(state, action_idx, reward, next_state, env.done)  # 存储经验

            state = next_state

            if len(agent.memory) > agent.batch_size:
                agent.replay(agent.batch_size)  # 训练DQN代理

        if e % 10 == 0:
            agent.save(f"iwanna-dqn-{e}.h5")  # 每10个episode保存一次模型
