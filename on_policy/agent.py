from typing import TYPE_CHECKING
import wandb
import gym
from tqdm import tqdm

from abc import ABC, abstractmethod

from utils.memory import RolloutBuffer

# WandB setup
if TYPE_CHECKING:
    from config.on_policy_config import GlobalConfig


class Agent(ABC):
    def __init__(self, config: 'GlobalConfig', env_id: str) -> None:
        """
        Generates an Agent that contains all of its variables, env etc...
        :param config: Config object that has all the infos necessary for the agent to learn
        """
        super().__init__()
        self.config = config

        # Generate the agent's environment
        self.env = gym.make(env_id)
        self.env_id = env_id

        # run params
        self.epochs = config['epochs']
        self.epoch_steps = config['epoch_steps']

    @abstractmethod
    def act(self, state: list) -> tuple:
        """
        Fetches an action from the agent, given an input
        :param state: observation vector given by the environment's .step() method. Must be given as received.
        """
        raise NotImplementedError("Agent.act must be defined in the sub-class")

    @abstractmethod
    def learn(self) -> dict:
        """
        Runs a learning step of the implemented RL agent, ie. action + learning on the networks
        :return: A dictionnary with the losses on this learning step
        """
        raise NotImplementedError(
            "Agent.learn must be defined in the sub-class")

    @abstractmethod
    def store(
        self,
        state: list,
        action: list,
        reward: float,
        done: bool,
        log_prob: list
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

    def train(self, render_rate: int = 20, log_to_wandb: bool = False):

        # wandb setup
        if log_to_wandb:
            wandb.init(
                project=f"{self.config['name']}_{self.env_id}", entity="thibaultlsdc")
            wandb.config.update(self.config)
        counter = tqdm(range(self.epochs), desc=f"Episode 0, Step 0/{self.epoch_steps}")
        for epoch in counter:
            state = self.env.reset()
            for step in range(self.epoch_steps):
                counter.set_description(
                    f"Episode {epoch}, Step {step+1}/{self.epoch_steps}")
                action, log_prob = self.act(state)
                next_state, reward, done, _ = self.env.step(action)

                self.store(state, action, reward, done, log_prob)

                state = next_state

                if done:
                    state = self.env.reset()

            metrics = self.learn()

            state = self.env.reset()
            self.env.render()

            total_reward = 0
            done = False
            n_steps = 0
            while not done:
                if epoch % render_rate == 0:
                    self.env.render()
                action, _ = self.act(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                n_steps += 1

            metrics['reward'] = total_reward
            metrics['ep_len'] = n_steps

            if log_to_wandb:
                wandb.log(metrics)
