import torch
import numpy as np
from copy import deepcopy

from torch import nn

from agents.agent import Agent
from utils.memory import BasicMemory
from agents.architectures import ModelLinear, ModelBounded

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import TD3Config


class TD3(Agent):
    def __init__(self, config: 'TD3Config') -> None:
        super(TD3, self).__init__(config)
        self.config = config

        self.name = config.name

        # replay memory
        self.memory = BasicMemory(config.capacity)

        # batch_size to sample from the memory
        self.batch_size = config.batch_size

        # torch gpu optimization
        self.device = config.device

        # discount
        self.gamma = config.gamma
        # polyak soft update
        self.tau = config.tau

        # action stddev
        self.std_start = config.std_start
        self.std_end = config.std_end
        self.std_decay = config.std_decay

        # policy training delay
        self.policy_delay = config.policy_delay

        # target smoothing and clipping
        self.target_std = config.target_std
        self.target_clipping = config.target_clipping

        # building model shapes
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = torch.tensor(
            self.env.action_space.high).to(self.device)

        critic_shape = [self.state_size + self.action_size] + \
            config.model_shape + [1]
        actor_shape = [self.state_size] + \
            config.model_shape + [self.action_size]

        # building models
        self.critic1 = ModelLinear(critic_shape).to(self.device)
        self.target_critic1 = deepcopy(self.critic1)

        self.critic2 = ModelLinear(critic_shape).to(self.device)
        self.target_critic2 = deepcopy(self.critic2)

        self.actor = ModelBounded(
            actor_shape, self.action_bound).to(self.device)
        self.target_actor = deepcopy(self.actor)

        # optimizers
        if config.optim['name'] == 'adam':
            self.actor_optim = torch.optim.Adam(
                self.actor.parameters(), config.optim['lr'])
            self.critic_optim = torch.optim.Adam(list(self.critic1.parameters(
            )) + list(self.critic2.parameters()), config.optim['lr'])
        elif config.optim['name'] == 'sgd':
            self.actor_optim = torch.optim.SGD(
                self.actor.parameters(), config.optim['lr'])
            self.critic_optim = torch.optim.SGD(list(self.critic1.parameters(
            )) + list(self.critic2.parameters()), config.optim['lr'])
        else:
            raise NotImplementedError(
                "Optimizer names should be in ['adam', 'sgd']")

    @property
    def action_std(self):
        return self.std_end + (self.std_start - self.std_end) * np.exp(- self.steps_trained / self.std_decay)

    def act(self, state, greedy: bool = False) -> list:
        """
        Get an action from the q_model, given the current state.
        :param state: input observation given by the environment
        :param greedy: boolean saying if the action should be noised or not, noised if false
        :return action: action command for the current state, to be given to the gym env, in a list
        """
        state = torch.tensor(state, device=self.device)

        with torch.no_grad():
            action = self.actor(state).squeeze()

            if not greedy:
                action = (action + torch.randn_like(action) * self.action_std *
                          self.action_bound).clamp(-self.action_bound, self.action_bound)
        return [x for x in action.cpu()]

    def learn(self):
        """
        Triggers one learning iteration and returns the los for the current step
        :return metrics: dictionnary containing all the metrics computed in the current step, for logs
        """
        if len(self.memory) < self.batch_size:
            return {}

        self.steps_trained += 1

        # sample batch from replay buffer
        transitions = self.memory.sample(self.batch_size)

        # from batch of transitions to transition of batches
        batch = self.memory.transition(*zip(*transitions))

        # # reward transform : dampening negative rewards for more daring agent
        # def reward_transform(x): return torch.max(x, 10*x)

        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.done)
        next_state = torch.cat(batch.next_state)

        # computing target
        with torch.no_grad():
            # target action with smoothing and clipping
            next_action = self.target_actor(next_state)
            next_action_noise = (torch.randn_like(
                action) * self.target_std).clamp(-self.target_clipping, self.target_clipping)
            next_action = (
                next_action + next_action_noise).clamp(-self.action_bound, self.action_bound)

            # target value from minimum of critic 1 and 2
            target_value1 = self.target_critic1(
                torch.cat([next_state, next_action], dim=-1))
            target_value2 = self.target_critic2(
                torch.cat([next_state, next_action], dim=-1))
            target_value = torch.min(target_value1, target_value2).squeeze()

            expected_value = reward + self.gamma * (1 - done) * target_value

        # compute Q values
        value1 = self.critic1(torch.cat([state, action], dim=-1)).squeeze()
        value2 = self.critic2(torch.cat([state, action], dim=-1)).squeeze()

        # compute Q losses
        criterion = nn.MSELoss()
        loss_q1 = criterion(value1, expected_value)
        loss_q2 = criterion(value2, expected_value)

        loss_critic = loss_q1 + loss_q2

        # critic optimization step
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        # policy delay
        if self.steps_trained % self.policy_delay == 0:
            # select action
            new_action = self.actor(state)
            # compute Q value to maximize
            loss_actor = - \
                self.critic1(torch.cat([state, new_action], dim=-1)).mean()

            # actor optimization step
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            # target updates
            for param, t_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                t_param.data.copy_(self.tau * t_param.data +
                                   (1 - self.tau) * param.data)
            for param, t_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                t_param.data.copy_(self.tau * t_param.data +
                                   (1 - self.tau) * param.data)
            for param, t_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                t_param.data.copy_(self.tau * t_param.data +
                                   (1 - self.tau) * param.data)

            return {"loss_ac": loss_actor, "loss_q1": loss_q1, "loss_q2": loss_q2}

        else:
            return {"loss_q1": loss_q1, "loss_q2": loss_q2}

    def save(self, state, action, reward, done, next_state):
        """
        Saves transition to the memory
        :args: all the informations of the transition, given by the env's step
        """
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor(
            [reward], device=self.device, dtype=torch.float32)
        done = torch.tensor([done], device=self.device, dtype=torch.float32)
        next_state = torch.tensor(next_state, device=self.device).unsqueeze(0)
        self.memory.store(state, action, reward, done, next_state)
