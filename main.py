from agents import DQN, DDPG, TD3
from config.config import DQNConfig, DDPGConfig, TD3Config

# agent = DQN(DQNConfig())
agent = DDPG(DDPGConfig())
# agent = TD3(TD3Config())

agent.train()
