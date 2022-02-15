import gym
import torch
import matplotlib.pyplot as plt

from itertools import count

# set up matplotlib
import matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    import IPython.display as display
plt.ion()

class Agent:
    def __init__(self, config) -> None:
        self.n_episodes = config.n_episodes

        self.episode_rewards = []

        self.plot = config.plot

        # initialize steps
        self.steps_done = 0

        self.env = gym.make(config.env_id)

        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')


    def make_episode(self, render=False):
        state = torch.tensor(self.env.reset(), device=self.device)
        rewards = 0
        for t in count():
            if render:
                self.env.render()
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action.item())

            rewards += reward

            next_state = torch.tensor(next_state, device=self.device)
            if done:
                next_state = None
            
            self.memory.store(state, action, reward, next_state)

            state = next_state

            loss = self.learn()
            print(f"Step {self.steps_done}, loss : {loss}, len mem : {len(self.memory)}", end='\r')

            if done:
                self.episode_rewards.append(rewards)
                if self.plot:
                    self.plot_rewards(50)
                break
        
    def plot_rewards(self, avg_size):
        plt.figure(3)
        plt.title('DQN')
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards, 'b')
        if len(self.episode_rewards) > avg_size:
            means = torch.tensor(self.episode_rewards).unfold(0, avg_size, 1).type(torch.float32).mean(1).view(-1)
            means = torch.cat((torch.zeros(avg_size-1), means))
            plt.plot(means.numpy())
        
        plt.pause(0.001)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def train(self):
        for episode in range(self.n_episodes):
            if episode % 1 == 0:
                self.make_episode(True)
            else:
                self.make_episode()

            if episode % self.conf.target_update == 0:
                self.target_model.load_state_dict(self.q_model.state_dict())
