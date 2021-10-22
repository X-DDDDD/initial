'''
Author: Whatever it takes
Date: 2021-10-22 21:13:58
LastEditTime: 2021-10-22 23:10:44
Description: 实现ADP
FilePath: \HCRL_ADP\adp_1.py
一份伏特加，加一点青柠，姜汁，啤酒，最重要的是，还有一点爱
'''
import gym
import numpy as np

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
agent = Qlearn(env.observation_space.n, env.action_space.n)

for step in range(10000):
    obs = env.reset()
    while True:
        # action = env.action_space.sample() # 写入action
        action = agent.sample(obs)
        # env.render()
        obs_prime, reward, done, info = env.step(action)
        agent.learn(obs_prime, obs, action, reward, done)
        obs = obs_prime
        if done:
            break
    print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(step+1, action, obs, reward, done, info))
env.close()