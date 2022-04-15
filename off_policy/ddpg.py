import torch
from torch import nn

from off_policy.agent import Agent
from utils.memory import BasicMemory
from utils.architectures import ModelLinear, ModelBounded
import numpy as np

from config.off_policy_config import DDPGConfig


class DDPG(Agent):
    def __init__(self, *args) -> None:
        super(DDPG, self).__init__(DDPGConfig, *args)
        self.conf = DDPGConfig

        self.memory = BasicMemory(DDPGConfig['capacity'])
        self.batch_size = DDPGConfig['batch_size']

        # env caracteristics
        self.obs_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.shape[0]

        self.ac_bounds = [self.env.action_space.low[0],
                          self.env.action_space.high[0]]

        # building full model shapes
        self.q_model_shape = [self.obs_size +
                              self.act_size] + DDPGConfig['model_shape'] + [1]
        self.ac_model_shape = [self.obs_size] + \
            DDPGConfig['model_shape'] + [self.act_size]

        # make q_model and q_target models and put them on selected device
        self.device = torch.device(
            DDPGConfig['device'] if torch.cuda.is_available() else 'cpu')
        self.q_model = ModelLinear(self.q_model_shape).to(self.device)
        self.q_target_model = ModelLinear(self.q_model_shape).to(self.device)

        # copying q_model's data into the target model
        self.q_target_model.load_state_dict(self.q_model.state_dict())

        # make ac and ac_target models and put them on selected device
        self.ac_model = ModelBounded(
            self.ac_model_shape, self.ac_bounds[1]).to(self.device)
        self.ac_target_model = ModelBounded(
            self.ac_model_shape, self.ac_bounds[1]).to(self.device)

        # copying ac_model's data into the target model
        self.ac_target_model.load_state_dict(self.ac_model.state_dict())

        # soft update parameter
        self.tau = DDPGConfig['tau']
        # discount factor
        self.gamma = DDPGConfig['gamma']

        # DDPGConfigure optimizers
        if DDPGConfig['optim'] == 'adam':
            self.q_optim = torch.optim.Adam(
                self.q_model.parameters(), DDPGConfig['lr'])
            self.ac_optim = torch.optim.Adam(
                self.ac_model.parameters(), DDPGConfig['lr'])
        elif DDPGConfig['optim'] == 'sgd':
            self.q_optim = torch.optim.SGD(
                self.q_model.parameters(), DDPGConfig['lr'])
            self.ac_optim = torch.optim.SGD(
                self.ac_model.parameters(), DDPGConfig['lr'])
        else:
            self.q_optim = torch.optim.Adam(self.q_model.parameters())
            self.ac_optim = torch.optim.Adam(self.ac_model.parameters())

        # exploration std parameters
        self.std_start = DDPGConfig['std_start']
        self.std_end = DDPGConfig['std_end']
        self.std_decay = DDPGConfig['std_decay']

    @property
    def std(self):
        """
        Just the decaying exploration std
        """
        return self.std_end + \
            (self.std_start - self.std_end) * \
            np.exp(- self.steps_trained / self.std_decay)

    def act(self, state, greedy=False):
        """
        Get an action from the q_model, given the current state.
        state : input observation given by the environment
        """
        state = torch.tensor(state, device=self.device)

        with torch.no_grad():
            action = self.ac_model(state).unsqueeze(0)
        if not greedy:
            action = (action + torch.randn_like(action) * self.std
                      ).clamp(self.ac_bounds[0], self.ac_bounds[1])

        return [x for x in action.squeeze(0).cpu()]

    def learn(self):
        """
        Triggers one learning iteration and returns the loss for the current step
        """
        if len(self.memory) < self.batch_size:
            return {"loss_q": 0}

        self.steps_trained += 1

        transitions = self.memory.sample(self.batch_size)

        batch = self.memory.transition(*zip(*transitions))

        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.done)
        next_state = torch.cat(batch.next_state)

        with torch.no_grad():
            next_action = self.ac_target_model(next_state)
            next_value = self.q_target_model(
                torch.cat([next_state, next_action], dim=-1)).squeeze(1)
            expected = reward + self.gamma * (1 - done) * next_value

        value = self.q_model(torch.cat([state, action], dim=-1)).squeeze(1)

        criterion = nn.MSELoss()

        loss_q = criterion(value, expected.type(torch.float32))

        self.q_optim.zero_grad()
        loss_q.backward()
        for param in self.q_model.parameters():
            param.grad.clamp_(-.1, .1)
        self.q_optim.step()

        new_action = self.ac_model(state)

        loss_ac = - \
            torch.mean(self.q_model(torch.cat([state, new_action], dim=-1)))

        self.ac_optim.zero_grad()
        loss_ac.backward()
        for param in self.ac_model.parameters():
            param.grad.clamp_(-.1, .1)
        self.ac_optim.step()

        for phi_target, phi in zip(self.q_target_model.parameters(), self.q_model.parameters()):
            phi_target.data.copy_(
                self.tau * phi_target.data + (1-self.tau) * phi.data)
        for phi_target, phi in zip(self.ac_target_model.parameters(), self.ac_model.parameters()):
            phi_target.data.copy_(
                self.tau * phi_target.data + (1-self.tau) * phi.data)

        return {"loss_q": loss_q.cpu().detach().item(), "loss_ac": loss_ac.cpu().detach().item()}

    def store(self, state, action, reward, done, next_state):
        """
        Saves transition to the memory
        """
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device, dtype=int)
        next_state = torch.tensor(next_state, device=self.device).unsqueeze(0)
        self.memory.store(state, action, reward, done, next_state)
