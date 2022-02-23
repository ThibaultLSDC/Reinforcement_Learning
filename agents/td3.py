import torch
import numpy as np

from torch import nn

from agents.agent import Agent
from utils.memory import Memory
from agents.architectures import ModelLinear

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import TD3Config


class TD3(Agent):
    def __init__(self, config: 'TD3Config') -> None:
        super(TD3, self).__init__(config)
        self.conf = config

        self.memory = Memory(config.capacity)

        self.obs_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.shape[0]

        self.q_model_shape = [self.obs_size +
                              self.act_size] + config.model_shape + [1]
        self.ac_model_shape = [self.obs_size] + \
            config.model_shape + [self.act_size]
        print(self.q_model_shape)
        print(self.ac_model_shape)

        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu')

        # make q1_model and q1_target models and put them on selected device
        self.q1_model = ModelLinear(self.q_model_shape).to(self.device)
        self.q1_target_model = ModelLinear(self.q_model_shape).to(self.device)
        # copying q_model's data into the target model
        self.q1_target_model.load_state_dict(self.q1_model.state_dict())

        # make q2_model and q2_target models and put them on selected device
        self.q2_model = ModelLinear(self.q_model_shape).to(self.device)
        self.q2_target_model = ModelLinear(self.q_model_shape).to(self.device)
        # copying q2_model's data into the target model
        self.q2_target_model.load_state_dict(self.q2_model.state_dict())

        # make ac and ac_target models and put them on selected device
        self.ac_model = ModelLinear(self.ac_model_shape).to(self.device)
        self.ac_target_model = ModelLinear(self.ac_model_shape).to(self.device)
        # copying ac_model's data into the target model
        self.ac_target_model.load_state_dict(self.ac_model.state_dict())

        self.update_method = config.update_method
        self.target_update = config.target_update

        self.tau = config.tau
        self.ac_smoothing = config.target_smoothing
        self.q_update = config.q_update_per_step
        self.target_std = config.target_std

        # configure q1_optimizer
        if config.optim['name'] == 'adam':
            self.q1_optim = torch.optim.Adam(
                self.q1_model.parameters(), config.optim['lr'])
        elif config.optim['name'] == 'sgd':
            self.q1_optim = torch.optim.SGD(
                self.q1_model.parameters(), config.optim['lr'])
        else:
            self.q1_optim = torch.optim.Adam(self.q_model.parameters())

        # configure q2_optimizer
        if config.optim['name'] == 'adam':
            self.q2_optim = torch.optim.Adam(
                self.q2_model.parameters(), config.optim['lr'])
        elif config.optim['name'] == 'sgd':
            self.q2_optim = torch.optim.SGD(
                self.q2_model.parameters(), config.optim['lr'])
        else:
            self.q2_optim = torch.optim.Adam(self.q_model.parameters())

        # configure ac_optimizer
        if config.optim['name'] == 'adam':
            self.ac_optim = torch.optim.Adam(
                self.ac_model.parameters(), config.optim['lr'])
        elif config.optim['name'] == 'sgd':
            self.ac_optim = torch.optim.SGD(
                self.ac_model.parameters(), config.optim['lr'])
        else:
            self.ac_optim = torch.optim.Adam(self.ac_model.parameters())

        self.ac_bounds = [self.env.action_space.low[0],
                          self.env.action_space.high[0]]

        self.std_start = config.std_start
        self.std_end = config.std_end
        self.std_decay = config.std_decay
        self.std = self.std_end + \
            (self.std_start - self.std_end) * \
            np.exp(- self.steps_done / self.std_decay)

    def act(self, state, greedy=False) -> list:
        """
        Get an action from the q_model, given the current state.
        state : input observation given by the environment
        """
        self.std = self.std_end + \
            (self.std_start - self.std_end) * \
            np.exp(- self.steps_done / self.std_decay)

        state = torch.tensor(state, device=self.device)

        with torch.no_grad():
            action = self.ac_model(state).unsqueeze(0)

        if not greedy:
            action = (action + self.std * torch.randn_like(action)
                      ).clamp(self.ac_bounds[0], self.ac_bounds[1])
        return [x for x in action.squeeze(0).cpu()]

    def learn(self):
        """
        Triggers one learning iteration and returns the los for the current step
        """
        self.steps_done += 1

        if len(self.memory) < self.conf.batch_size:
            return {"loss_q1": 0, "loss_q2": 0, "loss_ac": 0}

        for _ in range(self.q_update):
            transitions = self.memory.sample(self.conf.batch_size)

            batch = self.memory.transition(*zip(*transitions))

            non_final_mask = torch.tensor(
                [x is not None for x in batch.next_state])
            non_final_next_states = torch.cat(
                [x.unsqueeze(0).clone() for x in batch.next_state if x is not None])

            states = torch.cat([x.unsqueeze(0).clone() for x in batch.state])
            actions = torch.cat(batch.action, dim=0).to(
                self.device).unsqueeze(-1).squeeze(-1)
            rewards = torch.cat([torch.tensor(x).unsqueeze(0)
                                for x in batch.reward]).to(self.device)

            next_values1 = torch.zeros(
                self.conf.batch_size, device=self.device, dtype=torch.float32)
            next_values2 = torch.zeros(
                self.conf.batch_size, device=self.device, dtype=torch.float32)

            target_actions = self.ac_target_model(
                non_final_next_states).detach()

            target_noise = torch.clamp(torch.randn_like(
                target_actions) * torch.tensor(self.target_std), -self.ac_smoothing, self.ac_smoothing)

            target_actions = torch.clamp(
                target_noise + target_actions, self.ac_bounds[0], self.ac_bounds[1])

            target_input = torch.concat(
                [non_final_next_states, target_actions], dim=1)

            next_values1[non_final_mask] = self.q1_target_model(
                target_input).detach().squeeze()
            next_values2[non_final_mask] = self.q2_target_model(
                target_input).detach().squeeze()

            next_values = torch.cat([next_values1, next_values2], dim=-1)
            next_values = torch.min(next_values, dim=-1)[0]

            expected = next_values * self.conf.gamma + rewards

            values1 = self.q1_model(torch.concat([states, actions], dim=1))
            values2 = self.q2_model(torch.concat([states, actions], dim=1))

            criterion = nn.MSELoss()

            loss_q1 = criterion(values1.squeeze(),
                                expected.type(torch.float32))
            loss_q2 = criterion(values2.squeeze(),
                                expected.type(torch.float32))

            self.q1_optim.zero_grad()
            loss_q1.backward()
            for param in self.q1_model.parameters():
                param.grad.data.clamp_(-.1, .1)
            self.q1_optim.step()

            self.q2_optim.zero_grad()
            loss_q2.backward()
            for param in self.q2_model.parameters():
                param.grad.data.clamp_(-.1, .1)
            self.q2_optim.step()

        pred_actions = self.ac_model(states)
        loss_ac = - \
            torch.mean(self.q1_model(torch.concat(
                [states, pred_actions], dim=-1)))
        self.ac_optim.zero_grad()
        loss_ac.backward()
        for param in self.ac_model.parameters():
            param.grad.data.clamp_(-.1, .1)
        self.ac_optim.step()

        if self.update_method == 'periodic':
            if self.steps_done % self.target_update == 0:
                self.q1_target_model.load_state_dict(
                    self.q1_model.state_dict())
                self.q2_target_model.load_state_dict(
                    self.q2_model.state_dict())
                self.ac_target_model.load_state_dict(
                    self.ac_model.state_dict())

        elif self.update_method == 'soft':
            for phi_target, phi in zip(self.q1_target_model.parameters(), self.q1_model.parameters()):
                phi_target.data.copy_(
                    self.tau * phi_target.data + (1-self.tau) * phi.data)
            for phi_target, phi in zip(self.q2_target_model.parameters(), self.q2_model.parameters()):
                phi_target.data.copy_(
                    self.tau * phi_target.data + (1-self.tau) * phi.data)
            for phi_target, phi in zip(self.ac_target_model.parameters(), self.ac_model.parameters()):
                phi_target.data.copy_(
                    self.tau * phi_target.data + (1-self.tau) * phi.data)

        else:
            raise NotImplementedError(
                "Update method not implemented, 'periodic' and 'soft' are implemented for the moment")

        return {
            "loss_q1": loss_q1.cpu().detach().item(),
            "loss_q2": loss_q2.cpu().detach().item(),
            "loss_ac": loss_ac.cpu().detach().item()
        }

    def save(self, state, action, reward, next_state):
        """
        Saves transition to the memory
        """
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(
            next_state, device=self.device) if next_state is not None else None
        action = torch.tensor(action).unsqueeze(0)
        self.memory.store(state, action, reward, next_state)
