from typing import TYPE_CHECKING
import wandb
import gym
import matplotlib.pyplot as plt
import numpy as np

from itertools import count
from abc import ABC, abstractmethod
from utils.metrics import Metric

# set up matplotlib
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    import IPython.display as display
plt.ion()


# WandB setup
if TYPE_CHECKING:
    from config.config import GlobalConfig


class Agent(ABC):
    def __init__(self, config: 'GlobalConfig') -> None:
        """
        Generates an Agent that contains all of its variables
        :param config: Config object that has all the infos necessary for the agent to learn
        """
        super().__init__()
        self.config = config

        # Generate the agent's environment
        self.env = gym.make(config.env_id)

        # Metrics
        self.metrics_list = [
            f"loss_{x}" for x in self.config.losses] + ['reward']
        self.metrics = {key: Metric(key) for key in self.metrics_list}
        self.metrics['reward'] = Metric('reward', type='sum')

        # Duration of training, in number of episodes
        self.n_episodes = config.n_episodes

        # Logs and graphics
        # If one wants to plot the reward TODO: add possibility to plot other metrics
        self.plot = config.plot

        # wandb setup
        self.wandb = config.wandb
        if self.wandb:
            wandb.init(
                project=f"{config.name}_{config.env_id}", entity="thibaultlsdc")
            wandb.config.update(config.wandb_config)

        # initialize steps
        self.steps_done = 0

        # number of eps ran before training
        self.start_eps = config.start_eps

    @abstractmethod
    def act(self, state: list, greedy: bool) -> None:
        """
        Fetches an action from the agent, given an input
        :param state: observation vector given by the environment's .step() method. Must be given as received.

        """
        raise NotImplementedError("Agent.act must be defined in the sub-class")

    @abstractmethod
    def learn(self) -> dict:
        """
        Runs a learning step of the implemented RL agent, ie. action + learning on the networks
        :return: A dictionnary with the losses on this learning step, names must match the ones in self.metrics_list
        """
        raise NotImplementedError(
            "Agent.learn must be defined in the sub-class")

    @abstractmethod
    def save(
        self,
        state: list,
        action: list,
        reward: float,
        next_state: list
    ) -> None:
        """
        Saves the transition (state, action, reward, next_state) to the agent's memory, in the right format
        :param state: Observation from the environment
        :param action: Action given by act method
        :param reward: The reward given by the environment at a given step, given the state and action
        :param next_state: Resulting state after the action
        """
        raise NotImplementedError(
            "Agent.save must be defined in the sub-class")

    def make_episode(self, training: bool = True, render: bool = False, greedy: bool = False, plot: bool = False) -> None:
        """
        Runs a full episode in the agent's environment, until done is sent by the environment
        :param training: If the agent should run a learning session and improve its network(s). If False, the agent will run an episode with a greedy policy.
        :param render: True if the gym env should be displayed
        """
        if training:
            greedy = False
        state = self.env.reset()
        for t in count():
            if render:
                self.env.render()
            action = self.act(state, greedy)
            next_state, reward, done, _ = self.env.step(action)

            self.metrics['reward'].step(reward)

            self.save(state, action, reward, done, next_state)

            state = next_state

            if training:
                new_metrics = self.learn()
                for key in self.metrics_list:
                    if key != 'reward':
                        self.metrics[key].step(new_metrics[key])
                print(
                    f"Episode : {len(self.metrics['reward'].history)}, Step {self.steps_done}, Std : {self.eps:.4f}", end='\r')

            if done:
                if training:
                    for key in self.metrics_list:
                        self.metrics[key].new_ep()

                    if self.plot and plot:
                        self.plot_metric(self.metrics['reward'], 50)
                        # self.plot_metric(self.metrics['loss_q'], None)
                    if self.wandb:
                        desc = {key: self.metrics[key].history[-1]
                                for key in self.metrics_list}
                        wandb.log(desc)
                break

    @staticmethod
    def plot_metric(metrics: Metric, avg_size: int) -> None:
        """
        Pretty self telling, plots metric, with a sliding average of width avg_size

        :param metric: List containing the data to plot
        :param avg_size: if int, plots the sliding average of width avg_size, if None, no running average
        """
        metric = metrics.history[1:]  # NOTE : problem with the metrics
        plt.figure(metrics.name)
        plt.title(metrics.name)
        plt.xlabel('episodes')
        plt.ylabel(f"{metrics.name}")
        plt.plot(range(len(metric)), metric, 'b')
        if avg_size is not None:
            if len(metric) > avg_size:
                means = np.convolve(np.array(metric), np.ones(
                    avg_size)/avg_size, mode='valid')
                means = np.concatenate(
                    (np.ones(avg_size-1) * means[0], means), axis=0)
                plt.plot(means)

        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def train(self) -> None:
        """
        Self telling, trains the agent over n_episodes episodes
        """
        for ep in range(self.start_eps):
            self.make_episode(training=False, render=False, greedy=False)

        for episode in range(self.n_episodes):
            if episode % 1 == 0:
                if episode % 1 == 0:
                    self.make_episode(training=True, render=True, plot=True)
                else:
                    self.make_episode(training=True, render=True, plot=False)
            else:
                self.make_episode(training=True, render=False)

    def run(self, n_runs: int) -> None:
        """
        Runs n_runs runs of the agent in the env, with rendering
        :param n_runs: ...
        """
        for run in range(n_runs):
            self.make_episode(training=False, render=True, greedy=True)
