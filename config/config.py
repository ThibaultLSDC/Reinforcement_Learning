class GlobalConfig:
    a = 1

    # ID of the game to beat in gym
    env_id = 'CartPole-v1'

    # width of the models' hidden layers
    model_width = 64
    n_episodes = 600
    batch_size = 256

    plot = True
    # configuring wandb
    wandb = False
    wandb_config = {
    "model_width": model_width,
    "episodes": n_episodes,
    "batch_size": batch_size
    }

    # size of the memory
    capacity = 100000

    # torch device to use for the model
    device = 'cuda'
    # torch optimizer to be used
    optim = {
        'name':'adam',
        'lr':1e-5,
        }


class DQNConfig(GlobalConfig):
    
    name = 'dqn'

    eps_start = .9
    eps_end = .05
    # speed for the exponential decay
    eps_decay = 20000

    #
    gamma = 0.999
    # number of episodes before updating the target model
    target_update = 1

    update_method = 'soft'

    tau = .99