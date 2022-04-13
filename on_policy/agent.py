from typing import TYPE_CHECKING
import wandb
import gym
from tqdm import tqdm

from abc import ABC, abstractmethod

from utils.memory import SarsaBuffer

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

        # ReplayBuffer
        self.buffer = SarsaBuffer(config.buffer_capacity)

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
        :return: A dictionnary with the losses on this learning step
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

    def train(self, render_rate: int = 20, log_to_wandb: bool = False):

        # wandb setup
        if log_to_wandb:
            wandb.init(
                project=f"{self.config['name']}_{self.env_id}", entity="thibaultlsdc")
            wandb.config.update(self.config)

        # episodes counter
        n_episodes = 0

        # tqdm loop
        counter = tqdm(range(self.n_steps), desc='Pre-run phase')

        # initialize counts and env
        state = self.env.reset()
        episode_steps = 0
        total_reward = 0

        for step in counter:
            episode_steps += 1

            # render every few episodes
            if n_episodes % render_rate == 0:
                self.env.render()

            # by the end of the training, the agent is set to greedy
            greedy = (step > self.pre_run_steps + self.greedy_steps)
            action = self.act(state, greedy=greedy)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.save(state, action, reward, done, next_state, next_action)
            state = next_state

            # the agent learns at each step #TODO: train every n_steps
            new_metrics = self.learn()
            # tqdm description
            desc = f"Episode : {n_episodes}, Step {self.steps_trained}"
            counter.set_description(desc)

            if done:
                # iter counter, reset variables
                n_episodes += 1

                new_metrics['reward'] = total_reward

                state = self.env.reset()
                total_reward = 0
                episode_steps = 0

            if log_to_wandb:
                wandb.log(new_metrics)
