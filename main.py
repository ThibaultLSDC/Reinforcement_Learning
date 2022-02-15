from models.dqn import DQN
from config import Config

dqn = DQN(Config())

dqn.train()