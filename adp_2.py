'''
Author: Whatever it takes
Date: 2021-10-22 21:13:58
LastEditTime: 2021-10-23 15:30:23
Description: fail --- 实现ADP --- 由于env.step是不断向前推进的，所以不能用一个循环得到在一个固定的状态下做出所有选项的值
FilePath: \initial\adp_2.py
一份伏特加，加一点青柠，姜汁，啤酒，最重要的是，还有一点爱
'''
import gym
import numpy as np

# 添加ADP
class ADP(object):
    def __init__(self, obs_n, act_n, alpha=0.5, gamma = 0.9):
        self.obs_n = obs_n
        self.act_n = act_n
        self.q = np.zeros((obs_n, act_n))
        self.alpha = alpha
        self.gamma = gamma
    def learn(self, env, obs, act, reward, done):
        # 这里还有两个大循环
        # (1)遍历所有action_hat
        for action in range(env.action.space.n):
            # 得到所有的s_prime
            obs_prime, reward, done, info = env.step(action)   # 由于env.step是不断向前推进的，所以不能用一个循环得到在一个固定的状态下做出所有选项的值
            self.q[obs, action] += self.alpha * (reward + self.gamma * self.q[obs_prime, act]) # alpha 需要换成某种分布
        # (2)
        a_hat_star = agent.adp_sample(obs)
        # (3)用某种函数计算s_hat_prime --> 先换成网络 --> 本来是由系统决定 替换成env决定
        obs_hat_prime, _, _, _ = env.step(action)
        # (4)赋值下一步状态, 在采样路径 和 时间步骤 上继续循环

    def adp_sample(self):
        '''选择ADP算法的下一个状态'''
        # 先用贪婪策略选择
        pick_q = self.q[obs, :]
        maxq = np.max(pick_q)
        all_action = np.where(pick_q == maxq)[0] # 输出坐标 -- 对应相应的行为
        action = np.random.choice(all_action)
        return action
    def sample(self, obs):
        pick_q = self.q[obs, :]
        maxq = np.max(pick_q)
        all_action = np.where(pick_q == maxq)[0] # 输出坐标 -- 对应相应的行为
        action = np.random.choice(all_action)
        return action

# 需要添加控制行为的函数 本意期望ADP 先一行一行写Q-learning
class Qlearn(object):
    def __init__(self, obs_n, act_n, alpha=0.5, gamma = 0.9):
        self.obs_n = obs_n
        self.act_n = act_n
        self.q = np.zeros((obs_n, act_n))
        self.alpha = alpha
        self.gamma = gamma
    def learn(self, obs_prime, obs, act, reward, done):
        self.q[obs, act] += self.alpha * (reward + self.gamma * np.max(self.q[obs_prime]) - self.q[obs, act])
    def sample(self, obs):
        pick_q = self.q[obs, :]
        maxq = np.max(pick_q)
        all_action = np.where(pick_q == maxq)[0] # 输出坐标 -- 对应相应的行为
        action = np.random.choice(all_action)
        return action

env = gym.make('Taxi-v3')
# print(env.action_space.n) # 6
agent = ADP(env.observation_space.n, env.action_space.n)

for step in range(1):
    obs = env.reset()
    while True:
        # action = env.action_space.sample() # 写入action
        action = agent.sample(obs)
        # env.render()
        obs_prime, reward, done, info = env.step(action)
        agent.learn(env, obs, action, reward, done)
        obs = obs_prime
        if done:
            break
    print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(step+1, action, obs, reward, done, info))
env.close()