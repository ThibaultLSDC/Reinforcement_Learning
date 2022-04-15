import random as rd
from collections import deque, namedtuple


class BasicMemory:
    def __init__(self, capacity: int) -> None:
        """
        Basic deque memory
        :param capacity: maximum length of the memory
        """
        self.storage = deque([], maxlen=capacity)
        self.transition = namedtuple(
            'Transition',
            ('state', 'action', 'reward', 'done', 'next_state')
        )

    def store(self, *args) -> None:
        """
        Adds new elements to the memory, if length goes over capacity, then samples are erased (first in first out)
        """
        self.storage.append(self.transition(*args))

    def sample(self, batch_size: int) -> list:
        """
        Samples 'batch_size' elements in the memory at random.
        :param batch_size: number of elements to be fetched from the memory
        """
        return rd.sample(self.storage, batch_size)

    def __len__(self):
        return len(self.storage)


class RolloutBuffer:
    def __init__(self, capacity: int) -> None:
        """
        Buffer
        """
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.log_prob = []
        self.capacity = capacity

    def store(self, state, action, reward, done, log_prob) -> None:
        """
        Add new element
        """
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.log_prob.append(log_prob)

    def sample(self, batch_size: int) -> list:
        return rd.sample(self.storage, batch_size)

    def _len_(self):
        return len(self.state)

    def reset(self):
        self.__init__(self.capacity)

    def rollout(self, gamma, bootstrap):
        discount_rewards = []
        current_reward = gamma * bootstrap
        for reward, done in zip(reversed(self.reward), self.done):
            if done:
                current_reward = 0
            current_reward = reward + gamma * current_reward
            discount_rewards.insert(0, current_reward)
        
        self.reward = discount_rewards