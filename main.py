from agents import *
from config.config import *


agent = DDPG(DDPGConfig())

agent.run(1)
agent.train()
agent.run(10)