import torch
import random as rd
import numpy as np

from torch import nn

from agent import Agent
from memory import Memory


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.config import ConfigDQN


class Model(nn.Module): #TODO: virer ça
    def __init__(self, env, n=32) -> None:
        super(Model, self).__init__()
        self.core = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], n),
        nn.ReLU(),
        # nn.Linear(n, n),
        # nn.ReLU(),
        nn.Linear(n, env.action_space.n)
        )
    
    def forward(self, x):
        return self.core(x)



class DQN(Agent):
    def __init__(self, config : 'ConfigDQN') -> None:
        super(DQN, self).__init__(config)
        self.conf = config

        self.memory = Memory(config.capacity)

        # make q and target models and put them on selected device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.q_model = Model(self.env, config.model_width).to(self.device)
        self.target_model = Model(self.env, config.model_width).to(self.device)

        # copying q_model's data into the target model
        self.target_model.load_state_dict(self.q_model.state_dict())


        # configure optimizer
        if config.optim['name'] == 'adam':
            self.optim = torch.optim.Adam(self.q_model.parameters(), config.optim['lr'])
        elif config.optim['name'] == 'sgd':
            self.optim = torch.optim.SGD(self.q_model.parameters(), config.optim['lr'])
        else:
            self.optim = torch.optim.Adam(self.q_model.parameters())

    def act(self, state):
        """
        Get an action from the q_model, given the current state.
        state : input observation given by the environment
        """
        state = torch.tensor(state, device=self.device)

        eps_end = self.conf.eps_end
        eps_start = self.conf.eps_start
        eps_decay = self.conf.eps_decay

        threshold = rd.random()
        eps = eps_end + (eps_start - eps_end) * np.exp(-1 * self.steps_done / eps_decay)
        self.steps_done += 1

        if threshold > eps:
            with torch.no_grad():
                action = torch.argmax(self.q_model(state)).unsqueeze(0)

        else:
            action = torch.tensor([rd.randrange(self.env.action_space.n)], device=self.device, dtype=torch.int32)

        return action
    
    def learn(self):
        """
        Triggers one learning iteration and returns the los for the current step
        """
        if len(self.memory) < self.conf.batch_size:
            return 0
        
        transitions = self.memory.sample(self.conf.batch_size)

        batch = self.memory.transition(*zip(*transitions))

        non_final_mask = torch.tensor([x is not None for x in batch.next_state])
        non_final_next_states = torch.cat([x.unsqueeze(0).clone() for x in batch.next_state if x is not None])

        states = torch.cat([x.unsqueeze(0).clone() for x in batch.state])
        actions = torch.cat([x.clone() for x in batch.action]).type(torch.int64)
        rewards = torch.cat([torch.tensor(x).unsqueeze(0) for x in batch.reward]).to(self.device)

        values = self.q_model(states).gather(1, actions.unsqueeze(1))

        next_values = torch.zeros(self.conf.batch_size, device=self.device, dtype=torch.float32)
        next_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        expected = next_values * self.conf.gamma + rewards

        criterion = nn.MSELoss()

        loss = criterion(values.squeeze(), expected.type(torch.float32))

        self.optim.zero_grad()
        loss.backward()
        for param in self.q_model.parameters():
            param.grad.data.clamp_(-.1, .1)
        self.optim.step()
        return loss.cpu().detach().item()

    def save(self, state, action, reward, next_state):
        """
        Saves transition to the memory
        """
        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device) if next_state is not None else None
        self.memory.store(state, action, reward, next_state)