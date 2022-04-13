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


class SarsaBuffer:
    def __init__(self, capacity: int) -> None:
        """
        SARDSA buffer
        """
        self.storage = deque([], maxlen=capacity)
        self.transition = namedtuple(
            'Transition',
            ('state', 'action', 'reward', 'done', 'next_state', 'next_action')
        )

    def store(self, *args) -> None:
        """
        Add new element
        """
        self.storage.append(self.transition(*args))

    def sample(self, batch_size: int) -> list:
        return rd.sample(self.storage, batch_size)

    def _len_(self):
        return len(self.storage)

    def reset(self):
        self.__init__(self.storage.maxlen)
