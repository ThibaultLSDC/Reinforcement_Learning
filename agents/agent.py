import gym
import matplotlib.pyplot as plt
import numpy as np

from itertools import count
from abc import ABC, abstractmethod

# set up matplotlib
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    import IPython.display as display
plt.ion()


# WandB setup
import wandb


class Agent(ABC):
    def __init__(self, config) -> None:
        super().__init__()

        self.n_episodes = config.n_episodes

        self.episode_rewards = [] #TODO: better metrics
        self.episode_loss = []

        self.plot = config.plot
        self.wandb = config.wandb

        # initialize steps
        self.steps_done = 0

        self.env = gym.make(config.env_id)

        if self.wandb:
            wandb.init(project=f"{config.name}_{config.env_id}", entity="thibaultlsdc")
            wandb.config = config.wandb_config


    @abstractmethod
    def act(self, state):
        raise NotImplementedError("Agent.act must be defined in the sub-class")

    @abstractmethod
    def learn(self):
        raise NotImplementedError("Agent.learn must be defined in the sub-class")

    @abstractmethod
    def save(self, state, action, reward, next_state):
        raise NotImplementedError("Agent.save must be defined in the sub-class")

    def make_episode(self, training=True, render=False):
        greedy = not training
        state = self.env.reset()
        total_reward = 0
        total_loss = 0
        for t in count():
            if render:
                self.env.render()
            action = self.act(state, greedy)
            next_state, reward, done, _ = self.env.step(action)

            total_reward += reward

            next_state = next_state
            if done:
                next_state = None
            
            self.save(state, action, reward, next_state)

            state = next_state

            if training:
                total_loss += self.learn()
                print(f"Episode : {len(self.episode_loss)}, Step {self.steps_done}, loss : {total_loss / (t+1)}", end='\r')

            if done:
                if training:
                    self.episode_rewards.append(total_reward)
                    self.episode_loss.append(total_loss)
                    if self.plot:
                        self.plot_metric(self.episode_rewards, 50, 1)
                        # self.plot_metric(self.episode_loss, 50, 2)
                    if self.wandb:
                        wandb.log({'loss':total_loss / (len(self.episode_loss)+1), 'reward':total_reward})
                break
        
    @staticmethod
    def plot_metric(metric, avg_size, id):
        plt.figure(id)
        plt.title('DQN')
        plt.xlabel('episodes')
        plt.ylabel(f"{id}")
        plt.plot(range(len(metric)), metric, 'b')
        if len(metric) > avg_size:
            means = np.convolve(np.array(metric), np.ones(avg_size)/avg_size, mode='valid')
            means = np.concatenate((np.ones(avg_size-1) * means[0], means), axis=0)
            plt.plot(means)
        
        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def train(self):
        for episode in range(self.n_episodes):
            if episode % 1 == 0:
                self.make_episode(training=True, render=True)
            else:
                self.make_episode(training=True, render=False)
    
    def run(self, N):
        for ep in range(N):
            self.make_episode(training=False, render=True)