import torch
import random as rd
import numpy as np

from torch import nn

from agents.agent import Agent
from utils.memory import BasicMemory
from utils.architectures import ModelLinear
from config import DQNConfig


class DQN(Agent):
    def __init__(self, *args) -> None:
        super(DQN, self).__init__(DQNConfig, *args)
        self.conf = DQNConfig

        self.memory = BasicMemory(DQNConfig['capacity'])
        self.batch_size = DQNConfig['batch_size']

        self.q_model_shape = [self.env.observation_space.shape[0]
                              ] + DQNConfig['model_shape'] + [self.env.action_space.n]

        # make q and target models and put them on selected device
        self.device = torch.device(
            DQNConfig['device'] if torch.cuda.is_available() else 'cpu')
        self.q_model = ModelLinear(self.q_model_shape).to(self.device)
        self.target_model = ModelLinear(self.q_model_shape).to(self.device)

        # copying q_model's data into the target model
        self.target_model.load_state_dict(self.q_model.state_dict())

        self.update_method = DQNConfig['update_method']
        self.target_update = DQNConfig['target_update']

        self.gamma = DQNConfig['gamma']

        self.tau = DQNConfig['tau']

        # DQNConfigure optimizer
        if DQNConfig['optim'] == 'adam':
            self.optim = torch.optim.Adam(
                self.q_model.parameters(), DQNConfig['lr'])
        elif DQNConfig['optim'] == 'sgd':
            self.optim = torch.optim.SGD(
                self.q_model.parameters(), DQNConfig['lr'])
        else:
            self.optim = torch.optim.Adam(self.q_model.parameters())

        self.eps_start = DQNConfig['eps_start']
        self.eps_end = DQNConfig['eps_end']
        self.eps_decay = DQNConfig['eps_decay']

    def act(self, state, greedy=False):
        """
        Get an action from the q_model, given the current state.
        state : input observation given by the environment
        """
        state = torch.tensor(state, device=self.device)

        threshold = rd.random()
        self.std = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1 * self.steps_trained / self.eps_decay)

        if greedy:
            with torch.no_grad():
                action = torch.argmax(self.q_model(state)).unsqueeze(0)
        else:
            if threshold > self.std:
                with torch.no_grad():
                    action = torch.argmax(self.q_model(state)).unsqueeze(0)

            else:
                action = torch.tensor(
                    [rd.randrange(self.env.action_space.n)], device=self.device, dtype=torch.int32)

        return action.item()

    def learn(self):
        """
        Triggers one learning iteration and returns the los for the current step
        """

        if len(self.memory) < self.batch_size:
            return 0

        self.steps_trained += 1

        transitions = self.memory.sample(self.batch_size)

        batch = self.memory.transition(*zip(*transitions))

        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward) / 100
        done = torch.cat(batch.done)
        next_state = torch.cat(batch.next_state)

        with torch.no_grad():
            next_value = self.target_model(next_state).max(1)[0]
            expected = reward + (1 - done) * self.gamma * next_value

        value = self.q_model(state).gather(1, action.unsqueeze(1))

        criterion = nn.MSELoss()

        loss = criterion(value.squeeze(), expected)

        self.optim.zero_grad()
        loss.backward()
        for param in self.q_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        if self.update_method == 'periodic':
            if self.steps_trained % self.target_update == 0:
                self.target_model.load_state_dict(self.q_model.state_dict())
        elif self.update_method == 'soft':
            for phi_target, phi in zip(self.target_model.parameters(), self.q_model.parameters()):
                phi_target.data.copy_(
                    self.tau * phi_target.data + (1-self.tau) * phi.data)

        else:
            raise NotImplementedError(
                "Update method not implemented, 'periodic' and 'soft' are implemented for the moment")

        return {"loss_q": loss.cpu().detach().item()}

    def save(self, state, action, reward, done, next_state):
        """
        Saves transition to the memory
        """
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device, dtype=int)
        next_state = torch.tensor(next_state, device=self.device).unsqueeze(0)
        self.memory.store(state, action, reward, done, next_state)
