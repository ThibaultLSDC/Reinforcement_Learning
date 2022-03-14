import torch
import numpy as np
from copy import deepcopy

from torch import nn

from agents.agent import Agent
from utils.memory import BasicMemory
from utils.architectures import ModelBounded, TwinModel
from config import TD3Config


class TD3(Agent):
    def __init__(self, *args) -> None:
        super(TD3, self).__init__(TD3Config, *args)
        self.config = TD3Config

        self.name = TD3Config['name']

        # replay memory
        self.memory = BasicMemory(TD3Config['capacity'])

        # batch_size to sample from the memory
        self.batch_size = TD3Config['batch_size']

        # torch gpu optimization
        self.device = TD3Config['device']

        # discount
        self.gamma = TD3Config['gamma']
        # polyak soft update
        self.tau = TD3Config['tau']

        # action stddev
        self.std_start = TD3Config['std_start']
        self.std_end = TD3Config['std_end']
        self.std_decay = TD3Config['std_decay']

        # policy training delay
        self.policy_delay = TD3Config['policy_delay']

        # target smoothing and clipping
        self.target_std = TD3Config['target_std']
        self.target_clipping = TD3Config['target_clipping']

        # building model shapes
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = torch.tensor(
            self.env.action_space.high).to(self.device)

        critic_shape = [self.state_size + self.action_size] + \
            TD3Config['model_shape'] + [1]
        actor_shape = [self.state_size] + \
            TD3Config['model_shape'] + [self.action_size]

        # building models
        self.critic = TwinModel(critic_shape).to(self.device)
        self.target_critic = deepcopy(self.critic)

        self.actor = ModelBounded(
            actor_shape, self.action_bound).to(self.device)
        self.target_actor = deepcopy(self.actor)

        # optimizers
        if TD3Config['optim'] == 'adam':
            self.actor_optim = torch.optim.Adam(
                self.actor.parameters(), TD3Config['lr'])
            self.critic_optim = torch.optim.Adam(
                self.critic.parameters(), TD3Config['lr'])
        elif TD3Config['optim'] == 'sgd':
            self.actor_optim = torch.optim.SGD(
                self.actor.parameters(), TD3Config['lr'])
            self.critic_optim = torch.optim.SGD(
                self.critic.parameters(), TD3Config['lr'])
        else:
            raise NotImplementedError(
                "Optimizer names should be in ['adam', 'sgd']")

    @property
    def action_std(self):
        """
        Exploration std
        """
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
            target_value1, target_value2 = self.target_critic(
                torch.cat([next_state, next_action], dim=-1))
            target_value = torch.min(target_value1, target_value2).squeeze()

            expected_value = reward + self.gamma * (1 - done) * target_value

        # compute Q values
        value1, value2 = self.critic(
            torch.cat([state, action], dim=-1))

        # compute Q losses
        criterion = nn.MSELoss()
        loss_q1 = criterion(value1.squeeze(), expected_value)
        loss_q2 = criterion(value2.squeeze(), expected_value)

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
                self.critic.single(
                    torch.cat([state, new_action], dim=-1)).mean()

            # actor optimization step
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            # target updates
            for param, t_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                t_param.data.copy_(self.tau * t_param.data +
                                   (1 - self.tau) * param.data)
            for param, t_param in zip(self.critic.parameters(), self.target_critic.parameters()):
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
