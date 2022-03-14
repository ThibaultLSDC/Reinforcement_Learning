
from agents.dqn import DQN
from agents.ddpg import DDPG
from agents.td3 import TD3
from agents.sac import SAC

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-w', '--wandb', action='store_true',
                    help='Wether to upload the curves to wandb or not')
parser.add_argument('-e', '--env-id', type=str,
                    help='Environment name', default='CartPole-v1')
parser.add_argument('-m', '--model', type=str,
                    help='Model to build and train', default='dqn')
parser.add_argument('-r', '--render-rate', type=int,
                    help='Render every...', default=1)

args = parser.parse_args()

wandb = args.wandb
env_id = args.env_id
model_name = args.model
render_rate = args.render_rate

if model_name == 'dqn':
    agent = DQN(env_id)
elif model_name == 'ddpg':
    agent = DDPG(env_id)
elif model_name == 'td3':
    agent = TD3(env_id)
elif model_name == 'sac':
    agent = SAC(env_id)
else:
    raise NotImplementedError("Model not supported yet, try 'dqn' or 'ddpg'")

agent.train(render_rate=render_rate, log_to_wandb=wandb)
