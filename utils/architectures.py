from torch import nn, tanh, log
from torch.distributions import Normal


class ModelLinear(nn.Module):
    def __init__(self, data_sizes: list, zero_last: bool = True) -> None:
        super(ModelLinear, self).__init__()
        self.data_sizes = data_sizes
        for i in range(1, len(data_sizes)):
            setattr(self, f"layer_{i}", nn.Linear(
                data_sizes[i-1], data_sizes[i]))
            if not (i == len(data_sizes)-1):
                setattr(self, f"act_{i}", nn.ReLU())

        if zero_last:
            getattr(
                self, f"layer_{len(self.data_sizes)-1}").weight.data.fill_(0.)
            getattr(
                self, f"layer_{len(self.data_sizes)-1}").bias.data.fill_(0.)

    def forward(self, x):
        for i in range(1, len(self.data_sizes)):
            x = getattr(self, f"layer_{i}")(x)
            if not (i == len(self.data_sizes)-1):
                x = getattr(self, f"act_{i}")(x)
        return x


class ModelBounded(nn.Module):
    def __init__(self, data_sizes: list, output_amp, zero_last: bool = True) -> None:
        super(ModelBounded, self).__init__()
        self.data_sizes = data_sizes
        for i in range(1, len(data_sizes)):
            setattr(self, f"layer_{i}", nn.Linear(
                data_sizes[i-1], data_sizes[i]))
            if not (i == len(data_sizes)-1):
                setattr(self, f"act_{i}", nn.ReLU())

        self.output_amp = output_amp

        if zero_last:
            getattr(
                self, f"layer_{len(self.data_sizes)-1}").weight.data.fill_(0.)
            getattr(
                self, f"layer_{len(self.data_sizes)-1}").bias.data.fill_(0.)

    def forward(self, x):
        for i in range(1, len(self.data_sizes)):
            x = getattr(self, f"layer_{i}")(x)
            if not (i == len(self.data_sizes)-1):
                x = getattr(self, f"act_{i}")(x)
        return tanh(x) * self.output_amp


class TwinModel(nn.Module):
    def __init__(self, data_sizes: list, zero_last: bool = True) -> None:
        super(TwinModel, self).__init__()
        self.data_sizes = data_sizes
        self.model1 = ModelLinear(data_sizes, zero_last)
        self.model2 = ModelLinear(data_sizes, zero_last)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        return x1, x2

    def single(self, x):
        return self.model1(x)


class GaussianModel(nn.Module):
    def __init__(self, data_sizes: list, output_amp: float, min_log_std: float = -10., max_log_std: float = 2., zero_last: bool = True) -> None:
        super(GaussianModel, self).__init__()
        self.data_sizes = data_sizes
        for i in range(1, len(data_sizes)-1):
            setattr(self, f"layer_{i}", nn.Linear(
                data_sizes[i-1], data_sizes[i]))
            setattr(self, f"act_{i}", nn.ReLU())

        self.mu = nn.Linear(data_sizes[-2], data_sizes[-1])
        self.log_std = nn.Linear(data_sizes[-2], data_sizes[-1])

        self.output_amp = output_amp
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        if zero_last:
            self.mu.weight.data.fill_(0.)
            self.log_std.weight.data.fill_(0.)

    def forward(self, x):
        for i in range(1, len(self.data_sizes)-1):
            x = getattr(self, f"layer_{i}")(x)
            x = getattr(self, f"act_{i}")(x)

        mu = self.mu(x)
        log_std = self.log_std(x).clamp(self.min_log_std, self.max_log_std)
        return mu, log_std

    def sample(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()

        normal = Normal(mu, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action)

        action = tanh(action)
        # log_prob -= log(self.output_amp * (1 - action.pow(2)) + 1e-9)
        log_probs = log_prob - \
            log(self.output_amp * (1 - action.pow(2)) + 1e-9)

        log_probs = log_probs.sum(-1)

        return action * self.output_amp, log_probs, log_prob
