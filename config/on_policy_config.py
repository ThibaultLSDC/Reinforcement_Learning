GlobalConfig = {
    # width of the models' hidden layers
    'model_shape': [128],
    'n_steps': 1000000,
    'pre_run_steps': 100000,
    'greedy_steps': 900000,
    'batch_size': 64,

    # torch device to use for the model
    'device': 'cuda',
    # torch optimizer to be used
    'optim': 'adam',
    'lr': 3e-4,
}


PPOConfig = {
    'name': 'ppo',

    'gamma': .99,
    'alpha': 1,
    'beta': 1,

}
PPOConfig = PPOConfig | GlobalConfig
