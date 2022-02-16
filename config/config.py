class ConfigDQN: #TODO: GlobalConfig

    name = 'dqn'

    # ID of the game to beat in gym
    env_id = 'CartPole-v1'

    plot = False

    # configuring wandb
    wandb = True
    wandb_config = {
    "model_width": 64,
    "episodes": 2000,
    "batch_size": 256
    }

    # size of the memory
    capacity = 100000

    batch_size = 256
    n_episodes = 600

    eps_start = .9
    eps_end = .05
    # speed for the exponential decay
    eps_decay = 20000

    #
    gamma = 0.999
    # number of episodes before updating the target model
    target_update = 10

    # width of the models' hidden layers
    model_width = 64

    # torch device to use for the model
    device = 'cuda'
    # torch optimizer to be used
    optim = {
        'name':'aam',
        'lr':1e-5,
        }