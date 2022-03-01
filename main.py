from agents import DQN, DDPG, TD3, SAC
from config.config import DQNConfig, DDPGConfig, TD3Config, SACConfig

# agent = DQN(DQNConfig())
# agent = DDPG(DDPGConfig())
# agent = TD3(TD3Config())
agent = SAC(SACConfig())

agent.train(render_rate=1)
