import torch
from copy import deepcopy

from torch import nn

from agents.agent import Agent
from utils.memory import BasicMemory
from utils.architectures import TwinModel, GaussianModel
from config import SACConfig


class SAC(Agent):
    def __init__(self, *args) -> None:
        super(SAC, self).__init__(SACConfig, *args)
        self.config = SACConfig

        self.name = SACConfig['name']

        # replay memory
        self.memory = BasicMemory(SACConfig['capacity'])

        # batch_size to sample from the memory
        self.batch_size = SACConfig['batch_size']

        # torch gpu optimization
        self.device = SACConfig['device']

        # discount
        self.gamma = SACConfig['gamma']
        # polyak soft update
        self.tau = SACConfig['tau']
        # entropy weight
        self.alpha = SACConfig['alpha']
        # std clamp
        self.max_std = SACConfig['max_std']
        self.min_std = SACConfig['min_std']

        # building model shapes
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = torch.tensor(
            self.env.action_space.high).to(self.device)

        critic_shape = [self.state_size + self.action_size] + \
            SACConfig['model_shape'] + [1]
        actor_shape = [self.state_size] + \
            SACConfig['model_shape'] + [self.action_size]

        # building models
        self.critic = TwinModel(critic_shape).to(self.device)
        self.target_critic = deepcopy(self.critic)

        self.actor = GaussianModel(
            actor_shape, self.action_bound, self.min_std, self.max_std).to(self.device)

        # optimizers
        if SACConfig['optim'] == 'adam':
            self.actor_optim = torch.optim.Adam(
                self.actor.parameters(), SACConfig['lr'])
            self.critic_optim = torch.optim.Adam(
                self.critic.parameters(), SACConfig['lr'])
        elif SACConfig['optim'] == 'sgd':
            self.actor_optim = torch.optim.SGD(
                self.actor.parameters(), SACConfig['lr'])
            self.critic_optim = torch.optim.SGD(
                self.critic.parameters(), SACConfig['lr'])
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

        if not greedy:
            with torch.no_grad():
                action, _, _, _, _ = self.actor.sample(state)
        else:
            with torch.no_grad():
                _, _, _, action, _ = self.actor.sample(state)
        return [x for x in action.cpu()]

    def learn(self):
        """
        Triggers one learning iteration and returns the loss for the current step
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

        with torch.no_grad():
            # get sample action/log_prob from actor
            next_action, log_prob, _, mean, _ = self.actor.sample(next_state)

            # compute target value from sampled action
            target_value1, target_value2 = self.target_critic(
                torch.cat([next_state, next_action], dim=-1))
            target_value = torch.min(
                target_value1, target_value2).squeeze() - self.alpha * log_prob
            # compute expected value
            expected_value = reward + self.gamma * (1 - done) * target_value

        value1, value2 = self.critic(torch.cat([state, action], dim=-1))

        criterion = nn.MSELoss()

        loss_q1 = criterion(value1.squeeze(), expected_value)
        loss_q2 = criterion(value2.squeeze(), expected_value)

        loss_critic = loss_q1 + loss_q2

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        new_action, new_log_prob, logs, _, mean_std = self.actor.sample(state)

        new_value1, new_value2 = self.critic(
            torch.cat([state, new_action], dim=-1))
        new_value = torch.min(new_value1, new_value2).squeeze()

        loss_actor = (self.alpha * new_log_prob - new_value).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        for param, t_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            t_param.data.copy_(self.tau * t_param.data +
                               (1 - self.tau) * param.data)

        return {
            "loss_q1": loss_q1.detach().cpu(),
            "loss_q2": loss_q2.detach().cpu(),
            "loss_ac": loss_actor.detach().cpu(),
            "min_q_value": new_value.mean().detach().cpu(),
            "log_prob": new_log_prob.mean().detach().cpu(),
            "unsquashed_log_prob": logs.mean().detach().cpu(),
            "mean_std": mean_std.detach().cpu()
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
