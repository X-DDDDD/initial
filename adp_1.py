'''
Author: Whatever it takes
Date: 2021-10-22 21:13:58
LastEditTime: 2021-10-22 21:52:46
Description: 实现ADP
FilePath: \HCRL_ADP\adp_1.py
一份伏特加，加一点青柠，姜汁，啤酒，最重要的是，还有一点爱
'''
import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
for step in range(10000):
    action = env.action_space.sample() # 写入action
    env.render()
    obs, reward, done, info = env.step(action)
    print('step {}: action {}, obs {}, reward {}, done {}, info {}'.format(step, action, obs, reward, done, info))
env.close()