import torch
from copy import deepcopy

from torch import nn

from off_policy.agent import Agent
from utils.memory import BasicMemory
from utils.architectures import TwinModel, GaussianModel
from config.off_policy_config import SACConfig

from time import time


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

        if self.config['autotune']:
            self.log_alpha = torch.zeros(
                (1,), requires_grad=True, device=self.device)
            self.min_entropy = - \
                torch.prod(torch.Tensor(
                    self.env.action_space.shape).to(self.device)).item()
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha], lr=self.config['alpha_lr'])

        if self.config['eval']:
            try:
                self.load_model("/latest")
            except FileNotFoundError:
                self.save_model(0, 0)

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
                action, _, _, _, _ , _= self.actor.sample(state.float())
        else:
            with torch.no_grad():
                _, _, _, action, _, _ = self.actor.sample(state.float())
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
        top = time()
        transitions = self.memory.sample(self.batch_size)
        sample_time = time() - top

        # from batch of transitions to transition of batches
        top = time()
        batch = self.memory.transition(*zip(*transitions))
        batch_time = time() - top

        # # reward transform : dampening negative rewards for more daring agent
        # def reward_transform(x): return torch.max(x, 10*x)

        top = time()
        state = torch.cat(batch.state).to(self.device)
        action = torch.cat(batch.action).to(self.device)
        reward = torch.cat(batch.reward).to(self.device)
        done = torch.cat(batch.done).to(self.device)
        next_state = torch.cat(batch.next_state).to(self.device)
        cat_time = time() - top

        top = time()
        with torch.no_grad():
            # get sample action/log_prob from actor
            next_action, log_prob, _, mean, _, _ = self.actor.sample(
                next_state.float())

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

        value_learn_time = time() - top

        top = time()
        new_action, new_log_prob, logs, _, mean_std, _ = self.actor.sample(
            state.float())

        new_value1, new_value2 = self.critic(
            torch.cat([state, new_action], dim=-1))
        new_value = torch.min(new_value1, new_value2).squeeze()

        loss_actor = (self.alpha * new_log_prob - new_value).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        action_learn_time = time() - top

        if self.config['autotune']:
            top = time()
            alpha_loss = - (self.log_alpha * (new_log_prob +
                            self.min_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_time = time() - top

        top = time()
        for param, t_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            t_param.data.copy_(self.tau * t_param.data +
                               (1 - self.tau) * param.data)
        update_time = time() - top

        return {
            "loss_q1": loss_q1.detach().cpu(),
            "loss_q2": loss_q2.detach().cpu(),
            "loss_ac": loss_actor.detach().cpu(),
            "min_q_value": new_value.mean().detach().cpu(),
            "log_prob": new_log_prob.mean().detach().cpu(),
            "unsquashed_log_prob": logs.mean().detach().cpu(),
            "mean_std": mean_std.detach().cpu(),
            "alpha": self.alpha.detach().cpu().item(),
            "sample_time": sample_time,
            "batch_time": batch_time,
            "cat_time": cat_time,
            "value_learn_time": value_learn_time,
            "action_learn_time": action_learn_time,
            "alpha_time": alpha_time,
            "update_time": update_time
        }

    def store(self, state, action, reward, done, next_state):
        """
        Saves transition to the memory
        :args: all the informations of the transition, given by the env's step
        """
        state = torch.tensor(state, device='cpu').unsqueeze(0)
        action = torch.tensor(action, device='cpu').unsqueeze(0)
        reward = torch.tensor(
            [reward], device='cpu', dtype=torch.float32)
        done = torch.tensor([done], device='cpu', dtype=torch.float32)
        next_state = torch.tensor(next_state, device='cpu').unsqueeze(0)
        self.memory.store(state, action, reward, done, next_state)

    def get_state_dict(self, step, reward):
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_opt_state_dict': self.actor_optim.state_dict(),
            'critic_opt_state_dict': self.critic_optim.state_dict(),
            'reward': reward, 'step': step
        }

    def save_model(self, step, reward):

        state_dict = self.get_state_dict(step, reward)

        try:
            best_ckpt = torch.load(self.dir_save + "/best")
            if best_ckpt['reward'] < reward:
                print(f"Saving best model with {reward} reward at step {step}")
                torch.save(state_dict, self.dir_save + "/best")
        except FileNotFoundError:
            torch.save(state_dict, self.dir_save + "/best")

        torch.save(state_dict, self.dir_save + f"/{step}_steps")
        torch.save(state_dict, self.dir_save + "/latest")

    def load_model(self, path):
        loaded_state_dict = torch.load(self.dir_save + path)
        self.actor.load_state_dict(loaded_state_dict['actor_state_dict'])
        self.critic.load_state_dict(loaded_state_dict['critic_state_dict'])
        self.actor_optim.load_state_dict(
            loaded_state_dict['actor_opt_state_dict'])
        self.critic_optim.load_state_dict(
            loaded_state_dict['critic_opt_state_dict'])
        return loaded_state_dict['step']
