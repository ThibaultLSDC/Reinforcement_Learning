from torch import nn

class ModelLinear(nn.Module): #TODO: virer ça
    def __init__(self, env, n=32) -> None:
        super(ModelLinear, self).__init__()
        self.core = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], n),
        nn.ReLU(),
        # nn.Linear(n, n),
        # nn.ReLU(),
        nn.Linear(n, env.action_space.n)
        )
    
    def forward(self, x):
        return self.core(x)
