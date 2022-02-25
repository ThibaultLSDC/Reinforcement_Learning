class GlobalConfig:

    # ID of the game to beat in gym
    env_id = 'CartPole-v1'

    # width of the models' hidden layers
    model_shape = [64]
    n_episodes = 600
    batch_size = 128

    plot = True
    # configuring wandb
    wandb = False
    wandb_config = {
        "model_shape": model_shape,
        "episodes": n_episodes,
        "batch_size": batch_size
    }

    # size of the memory
    capacity = 100000

    # torch device to use for the model
    device = 'cuda'
    # torch optimizer to be used
    optim = {
        'name': 'adam',
        'lr': 1e-3,
    }

    losses = None

    start_eps = 20


class DQNConfig(GlobalConfig):

    name = 'dqn'

    eps_start = .9
    eps_end = .05
    # speed for the exponential decay
    eps_decay = 5000

    #
    gamma = 0.999
    # number of episodes before updating the target model
    target_update = 500

    update_method = 'soft'

    tau = .995

    losses = ['q']


class DDPGConfig(GlobalConfig):
    name = 'ddpg'

    gamma = .999

    target_update = 500
    update_method = 'soft'
    tau = .999

    std_start = 1
    std_end = .05
    std_decay = 20000

    losses = ['q', 'ac']

    def __init__(self) -> None:
        super().__init__()

        self.wandb_config['std_start'] = self.std_start
        self.wandb_config['std_end'] = self.std_end
        self.wandb_config['std_decay'] = self.std_decay
        self.wandb_config['tau'] = self.tau
        self.wandb_config['gamma'] = self.gamma


class TD3Config(GlobalConfig):
    name = 'td3'

    gamma = .99

    target_update = 500
    update_method = 'soft'
    tau = .995

    std_start = 1
    std_end = .05
    std_decay = 20000

    q_update_per_step = 1

    target_smoothing = 0.0

    target_std = 0.0

    losses = ['q1', 'q2', 'ac']

    def __init__(self) -> None:
        super().__init__()

        self.wandb_config['target_smoothing'] = self.target_smoothing
        self.wandb_config['std_start'] = self.std_start
        self.wandb_config['std_end'] = self.std_end
        self.wandb_config['std_decay'] = self.std_decay
        self.wandb_config['tau'] = self.tau
        self.wandb_config['gamma'] = self.gamma
        self.wandb_config['target_std'] = self.target_std
