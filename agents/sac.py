import torch
from copy import deepcopy

from torch import nn
import torch.distributions as td

from agents.agent import Agent
from utils.memory import BasicMemory
from agents.architectures import ModelLinear

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import SACConfig


class SAC(Agent):
    def __init__(self, config: 'SACConfig') -> None:
        super(SAC, self).__init__(config)
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
        # entropy weight
        self.alpha = config.alpha
        # std clamp
        self.max_std = config.max_std
        self.min_std = config.min_std

        # building model shapes
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = torch.tensor(
            self.env.action_space.high).to(self.device)

        critic_shape = [self.state_size + self.action_size] + \
            config.model_shape + [1]
        actor_shape = [self.state_size] + \
            config.model_shape + [2 * self.action_size]

        # building models
        self.critic1 = ModelLinear(critic_shape).to(self.device)
        self.target_critic1 = deepcopy(self.critic1)

        self.critic2 = ModelLinear(critic_shape).to(self.device)
        self.target_critic2 = deepcopy(self.critic2)

        self.actor = ModelLinear(actor_shape).to(self.device)

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

    def act(self, state, greedy: bool = False) -> list:
        """
        Get an action from the q_model, given the current state.
        :param state: input observation given by the environment
        :param greedy: not used here
        :return action: action command for the current state, to be given to the gym env, in a list
        """
        state = torch.tensor(state, device=self.device)

        with torch.no_grad():
            logits = self.actor(state)
            logits = self.actor(state).view(self.action_size, -1)
            action = self.action_bound * torch.tanh(
                td.Normal(logits[:, 0], torch.exp(logits[:, 1].clamp(self.min_std, self.max_std))).sample())
        return [x for x in action.cpu()]

    def learn(self):
        """
        Triggers one learning iteration and returns the los for the current step
        :return metrics: dictionnary containing all the metrics computed in the current step, for logs
        """
        if len(self.memory) < self.batch_size:
            return {}

        torch.autograd.set_detect_anomaly(True)

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

        with torch.no_grad():
            # get logits from actor
            logits = self.actor(next_state).view(
                self.batch_size, self.action_size, -1)
            # make distribution
            dist = td.Normal(logits[:, :, 0], torch.exp(
                logits[:, :, 1].clamp(self.min_std, self.max_std)))

            mean_std_logs = dist.stddev.mean()

            # get sample action, compute associated log probabilities
            next_action = dist.sample()
            log_prob = dist.log_prob(next_action).squeeze()
            next_action = torch.tanh(next_action)
            log_prob -= torch.log(self.action_bound *
                                  (1 - next_action.squeeze().pow(2)) + 1e-6)
            next_action = next_action * self.action_bound

            # compute target value from sampled action
            target_value1 = self.target_critic1(
                torch.cat([next_state, next_action], dim=-1))
            target_value2 = self.target_critic2(
                torch.cat([next_state, next_action], dim=-1))
            target_value = torch.min(target_value1, target_value2).squeeze()
            # compute expected value
            expected_value = reward + self.gamma * \
                (1 - done) * (target_value - self.alpha * log_prob)

        value1 = self.critic1(torch.cat([state, action], dim=-1)).squeeze()
        value2 = self.critic2(torch.cat([state, action], dim=-1)).squeeze()

        criterion = nn.MSELoss()

        loss_q1 = criterion(value1, expected_value)
        loss_q2 = criterion(value2, expected_value)

        loss_critic = loss_q1 + loss_q2

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        new_logits = self.actor(state).view(
            self.batch_size, self.action_size, -1)
        new_dist = td.Normal(
            new_logits[:, :, 0], torch.exp(new_logits[:, :, 1].clamp(self.min_std, self.max_std)))
        new_action = new_dist.rsample()
        new_log_prob = dist.log_prob(new_action).squeeze()
        new_action = torch.tanh(new_action)
        new_log_prob -= torch.log(self.action_bound *
                                  (1 - new_action.squeeze().pow(2)) + 1e-6)
        new_action = new_action * self.action_bound

        new_value1 = self.critic1(torch.cat([state, new_action], dim=-1))
        new_value2 = self.critic2(torch.cat([state, new_action], dim=-1))
        new_value = torch.min(new_value1, new_value2).squeeze()

        loss_actor = - (new_value - self.alpha * new_log_prob).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        for param, t_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            t_param.data.copy_(self.tau * t_param.data +
                               (1 - self.tau) * param.data)
        for param, t_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            t_param.data.copy_(self.tau * t_param.data +
                               (1 - self.tau) * param.data)

        return {
            "loss_q1": loss_q1.detach().cpu(),
            "loss_q2": loss_q2.detach().cpu(),
            "loss_ac": loss_actor.detach().cpu(),
            "mean_std": mean_std_logs.detach().cpu(),
            "min_q_value": new_value.mean().detach().cpu(),
            "log_prob": new_log_prob.mean().detach().cpu()
        }

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
