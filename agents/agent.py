from typing import TYPE_CHECKING
import wandb
import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        self.n_steps = config.n_steps
        self.pre_run_steps = config.pre_run_steps

        # Logs and graphics
        # If one wants to plot the reward
        self.plot = config.plot

        # wandb setup
        self.wandb = config.wandb
        if self.wandb:
            wandb.init(
                project=f"{config.name}_{config.env_id}", entity="thibaultlsdc")
            wandb.config.update(config.wandb_config)

        # initialize steps
        self.steps_trained = 0

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

    def train(self, render_rate: int = 100):

        state = self.env.reset()
        episode_steps = 0
        n_episodes = 0

        counter = tqdm(range(self.n_steps + self.pre_run_steps),
                       desc='Pre-run phase')

        for step in counter:
            training = (step > self.pre_run_steps)
            episode_steps += 1

            if n_episodes % render_rate == 0 and training:
                self.env.render()

            action = self.act(state, False)
            next_state, reward, done, _ = self.env.step(action)

            if training:
                self.metrics['reward'].step(reward)

            self.save(state, action, reward, done, next_state)

            state = next_state

            if training:
                new_metrics = self.learn()
                if type(new_metrics) == dict:
                    for key in self.metrics_list:
                        if key != 'reward':
                            self.metrics[key].step(new_metrics[key])
                        desc = f"Episode : {len(self.metrics['reward'].history)}, Step {self.steps_trained}, Std : {self.eps:.4f}"
                counter.set_description(desc)
            if done:
                if training:
                    n_episodes += 1

                    for key in self.metrics_list:
                        self.metrics[key].new_ep()

                    if self.plot:
                        self.plot_metric(self.metrics['reward'], 50)
                        # self.plot_metric(self.metrics['loss_q'], None)
                    if self.wandb:
                        desc = {key: self.metrics[key].history[-1]
                                for key in self.metrics_list}
                        wandb.log(desc)

                state = self.env.reset()
                episode_steps = 0
