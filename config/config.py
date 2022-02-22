class GlobalConfig:

    # ID of the game to beat in gym
    env_id = 'Pendulum-v1'

    # width of the models' hidden layers
    model_shape = [8, 8, 8]
    n_episodes = 600
    batch_size = 64

    plot = False
    # configuring wandb
    wandb = True
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


class DQNConfig(GlobalConfig):

    name = 'dqn'

    eps_start = .9
    eps_end = .05
    # speed for the exponential decay
    eps_decay = 10000

    #
    gamma = 0.999
    # number of episodes before updating the target model
    target_update = 500

    update_method = 'soft'

    tau = .999

    losses = ['q']


class DDPGConfig(GlobalConfig):
    name = 'ddpg'

    gamma = .999

    target_update = 500
    update_method = 'soft'
    tau = .999

    std_start = 1
    std_end = 1
    std_decay = 50000

    losses = ['q', 'ac']


class TD3Config(GlobalConfig):
    name = 'td3'

    gamma = .99

    target_update = 500
    update_method = 'soft'
    tau = .995

    q_update_per_step = 2

    std_start = .1
    std_end = .1
    std_decay = 50000

    target_smoothing = 0.5

    losses = ['q1', 'q2', 'ac']
