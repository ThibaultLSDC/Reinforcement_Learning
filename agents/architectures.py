from torch import nn, tanh


class ModelLinear(nn.Module):
    def __init__(self, data_sizes: list) -> None:
        super(ModelLinear, self).__init__()
        self.data_sizes = data_sizes
        for i in range(1, len(data_sizes)):
            setattr(self, f"layer_{i}", nn.Linear(
                data_sizes[i-1], data_sizes[i]))
            if not (i == len(data_sizes)-1):
                setattr(self, f"act_{i}", nn.ReLU())

    def forward(self, x):
        for i in range(1, len(self.data_sizes)):
            x = getattr(self, f"layer_{i}")(x)
            if not (i == len(self.data_sizes)-1):
                x = getattr(self, f"act_{i}")(x)
        return x


class ModelBounded(nn.Module):
    def __init__(self, data_sizes: list, output_amp) -> None:
        super(ModelBounded, self).__init__()
        self.data_sizes = data_sizes
        for i in range(1, len(data_sizes)):
            setattr(self, f"layer_{i}", nn.Linear(
                data_sizes[i-1], data_sizes[i]))
            if not (i == len(data_sizes)-1):
                setattr(self, f"act_{i}", nn.ReLU())

        self.output_amp = output_amp

    def forward(self, x):
        for i in range(1, len(self.data_sizes)):
            x = getattr(self, f"layer_{i}")(x)
            if not (i == len(self.data_sizes)-1):
                x = getattr(self, f"act_{i}")(x)
        return tanh(x) * self.output_amp
