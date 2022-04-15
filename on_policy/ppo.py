import numpy as np
import torch
from copy import deepcopy

from torch import nn, randperm

from on_policy.agent import Agent
from utils.memory import RolloutBuffer
from utils.architectures import ModelLinear, GaussianModel
from config.on_policy_config import PPOConfig

from time import time


class PPO(Agent):
    def __init__(self, *args) -> None:
        super(PPO, self).__init__(PPOConfig, *args)
        self.config = PPOConfig

        self.name = PPOConfig['name']

        # replay memory
        self.buffer = RolloutBuffer(PPOConfig['epoch_steps'])

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

        self.eps = PPOConfig['eps']

        # sub epochs
        self.sub_epochs = PPOConfig['sub_epochs']
        # normalize rewards
        self.normalize_rewards = PPOConfig['normalize_rewards']

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
            self.optim = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), PPOConfig['lr'])
        elif PPOConfig['optim'] == 'sgd':
            self.optim = torch.optim.SGD(list(self.actor.parameters()) + list(self.critic.parameters()), PPOConfig['lr'])
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
        action, log_probs, _, _, _, _ = self.actor.sample(torch.Tensor(state))
        return np.array(action), log_probs

    def learn(self):
        """
        Triggers one learning iteration and returns the loss for the current step
        :return metrics: dictionnary containing all the metrics computed in the current step, for logs
        """
        bootstrap_value = self.critic(self.buffer.state[-1]).detach().item()

        self.buffer.rollout(self.gamma, bootstrap_value)
        states = torch.cat(self.buffer.state).to(self.device)
        old_actions = torch.cat(self.buffer.action).to(self.device)
        rewards = torch.cat(self.buffer.reward).to(self.device)
        dones = torch.cat(self.buffer.done).to(self.device)
        old_log_probs = torch.cat(self.buffer.log_prob).to(self.device)

        if self.normalize_rewards:
            rewards = (rewards - rewards.mean()) / rewards.std()

        for epoch in range(self.sub_epochs):
            perm = randperm(states.size(0)).to(self.device)
            for i in range(perm.size(0) // self.batch_size):
                idx = perm[i*self.batch_size:(i+1)*self.batch_size]
                state = states[idx]
                old_action = old_actions[idx]
                reward = rewards[idx]
                done = dones[idx]
                old_log_prob = old_log_probs[idx]

                state_value = self.critic(state).squeeze()

                advantage = reward - state_value

                _, _, _, _, _, dist = self.actor.sample(state, train=True)
                log_prob = dist.log_prob(old_action)
                entropy = dist.entropy()

                ratio = log_prob - old_log_prob

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

                loss =  -torch.min(surr1, surr2) + self.alpha * nn.MSELoss()(state_value, reward) - self.beta * entropy

                self.optim.zero_grad()
                loss.mean().backward()
                self.optim.step()
        
        self.buffer.reset()

        return {
            'test': 1
        }       


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
        self.buffer.store(state, action, reward, done, log_prob)

    # def get_state_dict(self, step, reward):
    #     return {
    #         'actor_state_dict': self.actor.state_dict(),
    #         'critic_state_dict': self.critic.state_dict(),
    #         'actor_opt_state_dict': self.actor_optim.state_dict(),
    #         'critic_opt_state_dict': self.critic_optim.state_dict(),
    #         'reward': reward, 'step': step
    #     }

    # def save_model(self, step, reward):

    #     state_dict = self.get_state_dict(step, reward)

    #     try:
    #         best_ckpt = torch.load(self.dir_save + "/best")
    #         if best_ckpt['reward'] < reward:
    #             print(f"Saving best model with {reward} reward at step {step}")
    #             torch.save(state_dict, self.dir_save + "/best")
    #     except FileNotFoundError:
    #         torch.save(state_dict, self.dir_save + "/best")

    #     torch.save(state_dict, self.dir_save + f"/{step}_steps")
    #     torch.save(state_dict, self.dir_save + "/latest")

    # def load_model(self, path):
    #     loaded_state_dict = torch.load(self.dir_save + path)
    #     self.actor.load_state_dict(loaded_state_dict['actor_state_dict'])
    #     self.critic.load_state_dict(loaded_state_dict['critic_state_dict'])
    #     self.actor_optim.load_state_dict(
    #         loaded_state_dict['actor_opt_state_dict'])
    #     self.critic_optim.load_state_dict(
    #         loaded_state_dict['critic_opt_state_dict'])
    #     return loaded_state_dict['step']
