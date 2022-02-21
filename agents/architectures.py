from torch import nn

# class ModelLinear(nn.Module):
#     def __init__(self, env, n=32) -> None:
#         super(ModelLinear, self).__init__()
#         self.core = nn.Sequential(
#         nn.Linear(env.observation_space.shape[0], n),
#         nn.ReLU(),
#         # nn.Linear(n, n),
#         # nn.ReLU(),
#         nn.Linear(n, env.action_space.n)
#         )
#
#     def forward(self, x):
#         return self.core(x)


class ModelLinear(nn.Module):
    def __init__(self, data_sizes: list) -> None:
        super(ModelLinear, self).__init__()
        self.data_sizes = data_sizes
        for i in range(1, len(data_sizes)):
            setattr(self, f"layer_{i}", nn.Linear(data_sizes[i-1], data_sizes[i]))
            if not (i == len(data_sizes)-1):
                setattr(self, f"act_{i}", nn.ReLU())

    def forward(self, x):
        for i in range(1, len(self.data_sizes)):
            x = getattr(self, f"layer_{i}")(x)
            if not (i == len(self.data_sizes)-1):
                x = getattr(self, f"act_{i}")(x)
        return x
