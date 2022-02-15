import random as rd
from collections import deque, namedtuple


class Memory:
    def __init__(self, capacity : int) -> None:
        self.storage = deque([], maxlen=capacity)
        self.transition = namedtuple(
            'Transition',
            ('state', 'action', 'reward', 'next_state')
        )
    
    def store(self, *args):
        self.storage.append(self.transition(*args))
    
    def sample(self, batch_size):
        return rd.sample(self.storage, batch_size)

    def __len__(self):
        return len(self.storage)
