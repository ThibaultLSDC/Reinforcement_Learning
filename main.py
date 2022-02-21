from agents import DDPG
from config.config import DDPGConfig


agent = DDPG(DDPGConfig())

agent.run(1)
agent.train()
agent.run(10)
