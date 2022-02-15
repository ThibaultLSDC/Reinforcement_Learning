import torch
import random as rd
from collections import namedtuple, deque
import gym
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import count

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


Transition = namedtuple('Transition',
        ('state', 'action', 'reward', 'next_state'))


class Memory:
    def __init__(self, capacity) -> None:
        self.storage = deque([], maxlen=capacity)
    
    def store(self, *args):
        self.storage.append(Transition(*args))
    
    def sample(self, batch_size):
        return rd.sample(self.storage, batch_size)

    def __len__(self):
        return len(self.storage)


memory = Memory(100000)

env = gym.make('CartPole-v1')
# env = gym.make('LunarLander-v2')
# env = gym.make('Acrobot-v1')



class DQN(nn.Module):
    def __init__(self, n=32) -> None:
        super(DQN, self).__init__()
        self.core = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], n),
        nn.ReLU(),
        nn.Linear(n, n),
        nn.ReLU(),
        nn.Linear(n, env.action_space.n)
        )
    
    def forward(self, x):
        return self.core(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dqn = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(dqn.state_dict())

optimizer = torch.optim.Adam(dqn.parameters())

BATCH_SIZE = 256
EPISODES = 2000
EPS_START = .9
EPS_END = .05
EPS_DECAY = 5000
GAMMA = 0.999
TARGET_UPDATE = 10

steps_done = 0

def select_action(step, state):
    threshold = rd.random()
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1 * step / EPS_DECAY)
    step += 1
    if threshold > eps:
        with torch.no_grad():
            x = torch.argmax(dqn(state)).unsqueeze(0)
            return step, x, eps
    else:
        x = torch.tensor([rd.randrange(env.action_space.n)], device=device, dtype=torch.int32)
        return step, x, eps


episode_durations = []
episode_rewards = []


def plot_durations(avg_size):
    plt.figure(2)
    # plt.xlim((0, EPISODES))
    plt.title('DQN')
    plt.xlabel('episodes')
    plt.ylabel('durations')
    plt.plot(range(len(episode_durations)), episode_durations, 'b')
    if len(episode_durations) > avg_size:
        means = torch.tensor(episode_durations).unfold(0, avg_size, 1).type(torch.float32).mean(1).view(-1)
        means = torch.cat((torch.zeros(avg_size-1), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_rewards(avg_size):
    plt.figure(3)
    # plt.xlim((0, EPISODES))
    plt.title('DQN')
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.plot(range(len(episode_rewards)), episode_rewards, 'b')
    if len(episode_rewards) > avg_size:
        means = torch.tensor(episode_rewards).unfold(0, avg_size, 1).type(torch.float32).mean(1).view(-1)
        means = torch.cat((torch.zeros(avg_size-1), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor([x is not None for x in batch.next_state])
    non_final_next_states = torch.cat([x.unsqueeze(0).clone() for x in batch.next_state if x is not None])

    states = torch.cat([x.unsqueeze(0).clone() for x in batch.state])
    actions = torch.cat([x.clone() for x in batch.action]).type(torch.int64)
    rewards = torch.cat([torch.tensor(x).unsqueeze(0) for x in batch.reward]).to(device)

    values = dqn(states).gather(1, actions.unsqueeze(1))

    next_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32)
    next_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected = next_values * GAMMA + rewards

    criterion = nn.MSELoss()

    loss = criterion(values.squeeze(), expected.type(torch.float32))

    optimizer.zero_grad()
    loss.backward()
    for param in dqn.parameters():
        param.grad.data.clamp_(-.1, .1)
    optimizer.step()
    return loss.cpu().detach().item()


def make_episode(render=False):
    global steps_done
    state = torch.tensor(env.reset(), device=device)
    rewards = 0
    for t in count():
        if render:
            env.render()
        steps_done, action, eps = select_action(steps_done, state)
        next_state, reward, done, _ = env.step(action.item())

        if reward == -100:
            reward = -10

        rewards += reward

        next_state = torch.tensor(next_state, device=device)
        if done:
            next_state = None
        
        memory.store(state, action, reward, next_state)

        state = next_state

        loss = optimize()
        print(f"Step {steps_done}, eps : {eps:.4f}, loss : {loss}", end='\r')

        if done:
            episode_durations.append(t+1)
            episode_rewards.append(rewards)
            # plot_durations(50)
            plot_rewards(50)
            break

for episode in range(EPISODES):
    if episode % 1 == 0:
        make_episode(True)
    else:
        make_episode()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(dqn.state_dict())



print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()