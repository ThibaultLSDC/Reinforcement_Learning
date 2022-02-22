from agents import DDPG, TD3
from config.config import DDPGConfig, TD3Config

# agent = DDPG(DDPGConfig())
agent = TD3(TD3Config())

agent.run(1)
agent.train()
agent.run(10)
