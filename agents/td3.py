import torch
import random as rd
import numpy as np

from torch import nn

from agents.agent import Agent
from utils.memory import Memory
from agents.architectures import ModelLinear

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import DDPGConfig


class DDPG(Agent):
    def __init__(self, config : 'DDPGConfig') -> None:
        super(DDPG, self).__init__(config)
        self.conf = config

        self.memory = Memory(config.capacity)

        self.obs_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.shape[0]

        self.q_model_shape = [self.obs_size + self.act_size] + config.model_shape + [1]
        self.ac_model_shape = [self.obs_size] + config.model_shape + [1]

        # make q_model and q_target models and put them on selected device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.q_model = ModelLinear(self.q_model_shape).to(self.device)
        self.q_target_model = ModelLinear(self.q_model_shape).to(self.device)

        # copying q_model's data into the target model
        self.q_target_model.load_state_dict(self.q_model.state_dict())

        # make ac and ac_target models and put them on selected device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.ac_model = ModelLinear(self.ac_model_shape).to(self.device)
        self.ac_target_model = ModelLinear(self.ac_model_shape).to(self.device)

        # copying q_model's data into the target model
        self.ac_target_model.load_state_dict(self.ac_model.state_dict())

        self.update_method = config.update_method
        self.target_update = config.target_update

        self.tau = config.tau

        # configure q_optimizer
        if config.optim['name'] == 'adam':
            self.q_optim = torch.optim.Adam(self.q_model.parameters(), config.optim['lr'])
        elif config.optim['name'] == 'sgd':
            self.q_optim = torch.optim.SGD(self.q_model.parameters(), config.optim['lr'])
        else:
            self.q_optim = torch.optim.Adam(self.q_model.parameters())

        # configure ac_optimizer
        if config.optim['name'] == 'adam':
            self.ac_optim = torch.optim.Adam(self.ac_model.parameters(), config.optim['lr'])
        elif config.optim['name'] == 'sgd':
            self.ac_optim = torch.optim.SGD(self.ac_model.parameters(), config.optim['lr'])
        else:
            self.ac_optim = torch.optim.Adam(self.ac_model.parameters())
        
        self.ac_bounds = [self.env.action_space.low[0], self.env.action_space.high[0]]

        print(self.ac_bounds)

    def act(self, state, greedy=False):
        """
        Get an action from the q_model, given the current state.
        state : input observation given by the environment
        """
        state = torch.tensor(state, device=self.device)

        with torch.no_grad():
            action = self.ac_model(state).unsqueeze(0)
        if greedy:
            action = (action + torch.randn_like(action)).clamp_(self.ac_bounds[0], self.ac_bounds[1])

        return [action.item()]
    
    def learn(self):
        """
        Triggers one learning iteration and returns the los for the current step
        """
        self.steps_done += 1

        if len(self.memory) < self.conf.batch_size:
            return 0
        
        transitions = self.memory.sample(self.conf.batch_size)

        batch = self.memory.transition(*zip(*transitions))

        non_final_mask = torch.tensor([x is not None for x in batch.next_state])
        non_final_next_states = torch.cat([x.unsqueeze(0).clone() for x in batch.next_state if x is not None])

        states = torch.cat([x.unsqueeze(0).clone() for x in batch.state])
        actions = torch.tensor([x[0] for x in batch.action]).to(self.device).unsqueeze(-1)
        rewards = torch.cat([torch.tensor(x).unsqueeze(0) for x in batch.reward]).to(self.device)

        values = self.q_model(torch.concat([states, actions], dim=1))

        next_values = torch.zeros(self.conf.batch_size, device=self.device, dtype=torch.float32)

        target_actions = self.ac_target_model(non_final_next_states).detach()
        target_input = torch.concat([non_final_next_states, target_actions], dim=1)

        next_values[non_final_mask] = self.q_target_model(target_input).detach().squeeze()

        expected = next_values * self.conf.gamma + rewards

        criterion = nn.MSELoss()

        loss1 = criterion(values.squeeze(), expected.type(torch.float32))

        self.q_optim.zero_grad()
        loss1.backward()
        for param in self.q_model.parameters():
            param.grad.data.clamp_(-.1, .1)
        self.q_optim.step()

        pred_actions = self.ac_model(states)
        loss2 = -torch.mean(self.q_model(torch.concat([states, pred_actions], dim=-1)))
        self.ac_optim.zero_grad()
        loss2.backward()
        for param in self.q_model.parameters():
            param.grad.data.clamp_(-.1, .1)
        self.ac_optim.step()

        if self.update_method == 'periodic':
            if self.steps_done % self.target_update == 0:
                self.q_target_model.load_state_dict(self.q_model.state_dict())
                self.ac_target_model.load_state_dict(self.ac_model.state_dict())
        elif self.update_method == 'soft':
            for phi_target, phi in zip(self.q_target_model.parameters(), self.q_model.parameters()):
                phi_target.data.copy_(self.tau * phi_target.data + (1-self.tau) * phi.data)
            for phi_target, phi in zip(self.ac_target_model.parameters(), self.ac_model.parameters()):
                phi_target.data.copy_(self.tau * phi_target.data + (1-self.tau) * phi.data)
            

        else:
            raise NotImplementedError("Update method not implemented, 'periodic' and 'soft' are implemented for the moment")

        return loss1.cpu().detach().item() + loss2.cpu().detach().item()

    def save(self, state, action, reward, next_state):
        """
        Saves transition to the memory
        """
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device) if next_state is not None else None
        self.memory.store(state, action, reward, next_state)