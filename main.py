from models.dqn import DQN
from config.config import ConfigDQN


dqn = DQN(ConfigDQN())

dqn.train()