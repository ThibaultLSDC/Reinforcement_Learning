class Metric:
    def __init__(self, name: str) -> None:
        """
        Generate a metric tracker, that computes the average of the metric obtained during an episode at each step
        :param name: the name of the said metric
        """
        self.history = []
        self.current_step = 0
        self.name = name
        self.value = 0

    def step(self, value: float) -> None:
        """
        Adds a value to the running average of the running episode
        :param value: value to add
        """
        self.value = (self.value * self.current_step + value) / \
            (self.current_step + 1)
        self.current_step += 1

    def new_ep(self) -> None:
        """
        Appends the running average of the episode to the history list, resets steps and values
        """
        self.history.append(self.value)
        self.value = 0
        self.current_step = 0

    def reset(self):
        """
        Self telling
        """
        self.__init__(self.name)
