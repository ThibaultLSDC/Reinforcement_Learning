import torch
from copy import deepcopy

from torch import nn

from on_policy.agent import Agent
from utils.memory import Buffer
from utils.architectures import ModelLinear, GaussianModel
from config.on_policy_config import PPOConfig

from time import time


class SAC(Agent):
    def __init__(self, *args) -> None:
        super(SAC, self).__init__(PPOConfig, *args)
        self.config = PPOConfig

        self.name = PPOConfig['name']

        # replay memory
        self.buffer = Buffer(PPOConfig['capacity'])

        # batch_size to sample from the memory
        self.batch_size = PPOConfig['batch_size']

        # torch gpu optimization
        self.device = PPOConfig['device']

        # discount
        self.gamma = PPOConfig['gamma']
        # value weight
        self.alpha = PPOConfig['alpha']
        # entropy weight
        self.beta = PPOConfig['beta']

        # building model shapes
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = torch.tensor(
            self.env.action_space.high).to(self.device)

        critic_shape = [self.state_size] + \
            PPOConfig['model_shape'] + [1]
        actor_shape = [self.state_size] + \
            PPOConfig['model_shape'] + [self.action_size]

        # building models
        self.critic = ModelLinear(critic_shape).to(self.device)

        self.actor = GaussianModel(
            actor_shape, self.action_bound, self.config['min_log_std'], self.config['max_log_std']).to(self.device)

        # optimizers
        if PPOConfig['optim'] == 'adam':
            self.actor_optim = torch.optim.Adam(
                self.actor.parameters(), PPOConfig['lr'])
            self.critic_optim = torch.optim.Adam(
                self.critic.parameters(), PPOConfig['lr'])
        elif PPOConfig['optim'] == 'sgd':
            self.actor_optim = torch.optim.SGD(
                self.actor.parameters(), PPOConfig['lr'])
            self.critic_optim = torch.optim.SGD(
                self.critic.parameters(), PPOConfig['lr'])
        else:
            raise NotImplementedError(
                "Optimizer names should be in ['adam', 'sgd']")

        # if self.config['eval']:
        #     try:
        #         self.load_model("/latest")
        #     except FileNotFoundError:
        #         self.save_model(0, 0)

    def act(self, state) -> list:
        """
        Get an action from the q_model, given the current state.
        :param state: input observation given by the environment
        :param greedy: not used here
        :return action: action command for the current state, to be given to the gym env, in a list
        """
        action, log_probs, _, _, _ = self.actor.sample(state)
        return action, log_probs

    def learn(self):
        """
        Triggers one learning iteration and returns the loss for the current step
        :return metrics: dictionnary containing all the metrics computed in the current step, for logs
        """
        pass

    def store(self, state, action, reward, done, log_prob):
        """
        Saves transition to the memory
        :args: all the informations of the transition, given by the env's step
        """
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action = torch.tensor(action, device=self.device).unsqueeze(0)
        reward = torch.tensor(
            [reward], device=self.device, dtype=torch.float32)
        done = torch.tensor([done], device=self.device, dtype=torch.float32)
        log_prob = log_prob.to(self.device).unsqueeze(0)
        self.memory.store(state, action, reward, done, log_prob)

    def get_state_dict(self, step, reward):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_opt_state_dict': self.actor_optim.state_dict(),
            'critic_opt_state_dict': self.critic_optim.state_dict(),
            'reward': reward, 'step': step
        }

    def save_model(self, step, reward):

        state_dict = self.get_state_dict(step, reward)

        try:
            best_ckpt = torch.load(self.dir_save + "/best")
            if best_ckpt['reward'] < reward:
                print(f"Saving best model with {reward} reward at step {step}")
                torch.save(state_dict, self.dir_save + "/best")
        except FileNotFoundError:
            torch.save(state_dict, self.dir_save + "/best")

        torch.save(state_dict, self.dir_save + f"/{step}_steps")
        torch.save(state_dict, self.dir_save + "/latest")

    def load_model(self, path):
        loaded_state_dict = torch.load(self.dir_save + path)
        self.actor.load_state_dict(loaded_state_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_state_dict['critic_state_dict'])
        self.actor_optim.load_state_dict(
            loaded_state_dict['actor_opt_state_dict'])
        self.critic_optim.load_state_dict(
            loaded_state_dict['critic_opt_state_dict'])
        return loaded_state_dict['step']
