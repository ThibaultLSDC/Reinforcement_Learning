from agents.ddpg import DDPG
# from agents.dqn import DQN
from agents.dqn_new_memory import DQN
from agents.td3 import TD3

if __name__ == '__main__':
    agent = DQN()
    agent = DDPG()
    agent = TD3()
