class GlobalConfig:

    # ID of the game to beat in gym
    env_id = 'MountainCarContinuous-v0'

    # width of the models' hidden layers
    model_shape = [256]
    n_steps = 400000
    pre_run_steps = 100000
    greedy_steps = 380000
    batch_size = 256

    plot = False
    # configuring wandb
    wandb = True
    wandb_config = {
        "model_shape": model_shape,
        "steps": n_steps,
        "prerun_steps": pre_run_steps,
        "greedy_steps": greedy_steps,
        "batch_size": batch_size
    }

    # size of the memory
    capacity = 500000

    # torch device to use for the model
    device = 'cuda'
    # torch optimizer to be used
    optim = {
        'name': 'adam',
        'lr': 1e-4,
    }

    losses = None


class DQNConfig(GlobalConfig):

    name = 'dqn'

    eps_start = .5
    eps_end = .05
    # speed for the exponential decay
    eps_decay = 4000

    # discount factor
    gamma = 0.99
    # number of episodes before updating the target model
    target_update = 1000

    update_method = 'soft'

    # for soft update
    tau = .995

    losses = ['q']

    def __init__(self) -> None:
        super().__init__()

        self.wandb_config['eps_start'] = self.eps_start
        self.wandb_config['eps_end'] = self.eps_end
        self.wandb_config['eps_decay'] = self.eps_decay
        self.wandb_config['tau'] = self.tau
        self.wandb_config['gamma'] = self.gamma


class DDPGConfig(GlobalConfig):
    name = 'ddpg'

    gamma = .99

    target_update = 500
    update_method = 'soft'
    tau = .995

    std_start = .9
    std_end = .1
    std_decay = 10000

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

    # discount value
    gamma = .99

    # soft update rate
    tau = .995

    # action std and decay
    std_start = .1
    std_end = .1
    std_decay = 10000

    policy_delay = 2

    # target smoothing
    target_std = 0.2
    target_clipping = 0.5

    losses = ['q1', 'q2', 'ac']

    def __init__(self) -> None:
        super().__init__()

        self.wandb_config['target_smoothing'] = self.target_std
        self.wandb_config['std_start'] = self.std_start
        self.wandb_config['std_end'] = self.std_end
        self.wandb_config['std_decay'] = self.std_decay
        self.wandb_config['tau'] = self.tau
        self.wandb_config['gamma'] = self.gamma
        self.wandb_config['target_std'] = self.target_std
        self.wandb_config['policy_delay'] = self.policy_delay
