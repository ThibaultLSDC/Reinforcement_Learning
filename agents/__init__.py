from agents.ddpg import DDPG
from agents.dqn import DQN
from agents.td3 import TD3
from agents.sac import SAC

if __name__ == '__main__':
    agent = DQN()
    agent = DDPG()
    agent = TD3()
