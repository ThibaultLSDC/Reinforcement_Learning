from agents.dqn import DQN
from config.config import DQNConfig


dqn = DQN(ConfigDQN())

dqn.train()