from typing import TYPE_CHECKING
from lockfile import AlreadyLocked
import wandb
import gym
from tqdm import tqdm
import os as os

from abc import ABC, abstractmethod

# WandB setup
if TYPE_CHECKING:
    from config.off_policy_config import GlobalConfig

from time import time


class Agent(ABC):
    def __init__(self, config: 'GlobalConfig', env_id: str, run_id: str) -> None:
        """
        Generates an Agent that contains all of its variables, env etc...
        :param config: Config object that has all the infos necessary for the agent to learn
        """
        super().__init__()
        self.config = config

        # Generate the agent's environment
        self.env = gym.make(env_id)
        self.env_id = env_id

        # Duration of training, in number of episodes
        self.n_steps = config['n_steps']
        self.pre_run_steps = config['pre_run_steps']
        self.greedy_steps = config['greedy_steps']

        # initialize steps
        self.steps_trained = 0

        # save folder
        self.dir_save = f"./models/{config['name']}/{env_id}/{run_id}"
        self.run_id = run_id
        # assert (not os.path.exists(self.dir_save)), "run_id already taken"
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)

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
    def store(
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
            "Agent.store must be defined in the sub-class")
    
    @abstractmethod
    def save_model(
        self,
        step,
        reward
    ):
        """
        Saves model checkpoints to path
        """
        raise NotImplementedError("Agent.save_model must be defined in the sub-class")

    @abstractmethod
    def load_model(
        self,
        path
    ):
        """
        Loads model checkpoints from path
        """
        raise NotImplementedError("Agent.load_model must be defined in the sub-class")

    def train(self, render_rate: int = 20, log_to_wandb: bool = False):
        
        if os.path.exists(self.dir_save) and self.config['eval']:
            print('Starting from already trained model')
            self.steps_trained += self.load_model("/latest")
        # wandb setup
        if log_to_wandb:
            wandb.init(
                project=f"{self.config['name']}_{self.env_id}", entity="thibaultlsdc", name=self.run_id)
            wandb.config.update(self.config)

        # tqdm loop
        counter = tqdm(range(self.pre_run_steps), desc='Pre-run phase')

        # pre-run steps
        state = self.env.reset()
        for step in counter:
            # sample random actions
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            # save the random transitions
            self.store(state, action, reward, done, next_state)
            state = next_state
            if done:
                state = self.env.reset()

        print(f"Sampled {self.pre_run_steps} steps, starting training...")

        # episodes counter
        n_episodes = 0

        # tqdm loop
        counter = tqdm(range(self.n_steps), desc="Episode : 0, Step 0")

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
            greedy = (step > self.n_steps - self.greedy_steps)

            top = time()
            action = self.act(state, greedy=greedy)
            act_time = time() - top

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            top = time()
            self.store(state, action, reward, done, next_state)
            save_time = time() - top

            state = next_state

            # the agent learns at each step #TODO: train every n_steps
            top = time()
            new_metrics = self.learn()
            learn_time = time() - top

            # timers
            new_metrics["act_time"] = act_time
            new_metrics["save_time"] = save_time
            new_metrics["learn_time"] = learn_time

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
            
            if self.config['eval'] and self.steps_trained % self.config['eval_rate'] == 0:
                self.eval()

    def eval(self):

        done = False
        tmp_env = gym.make(self.env_id)
        state = tmp_env.reset()
        total_reward = 0

        while not done:
            action = self.act(state, greedy=True)
            state, reward, done, _ = tmp_env.step(action)
            total_reward += reward

        self.save_model(self.steps_trained, total_reward)