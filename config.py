class Config:

    # ID of the game to beat in gym
    env_id = 'CartPole-v1'

    plot = True

    # size of the memory
    capacity = 100000

    batch_size = 256
    n_episodes = 2000

    eps_start = .9
    eps_end = .05
    # speed for the exponential decay
    eps_decay = 20000

    #
    gamma = 0.999
    # number of episodes before updating the target model
    target_update = 10

    # width of the models' hidden layers
    model_width = 32

    # torch device to use for the model
    device = 'cuda'
    # torch optimizer to be used
    optim = {
        'name':'aam',
        'lr':1e-5,
        }