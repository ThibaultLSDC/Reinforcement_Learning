GlobalConfig = {
    # width of the models' hidden layers
    'model_shape': [128],
    'epochs': 10000,
    'epoch_steps': 4096,
    'batch_size': 64,

    # torch device to use for the model
    'device': 'cpu',
    # torch optimizer to be used
    'optim': 'adam',
    'lr': 3e-4,
}


PPOConfig = {
    'name': 'ppo',

    'gamma': .99,
    'alpha': 1,
    'beta': 1,
    'eps': .01,

    'sub_epochs': 10,
    'normalize_rewards':True,

    'min_log_std': -5,
    'max_log_std': 2,
}
PPOConfig = PPOConfig | GlobalConfig
