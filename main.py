from agents.dqn import DQN
from config.config import DQNConfig


dqn = DQN(DQNConfig())

dqn.run(10)
dqn.train()
dqn.run(10)