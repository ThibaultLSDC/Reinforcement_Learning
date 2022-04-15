GlobalConfig = {
    # width of the models' hidden layers
    'model_shape': [128],
    'n_steps': 3000000,
    'pre_run_steps': 5000,
    'greedy_steps': 100000,
    'batch_size': 128,

    # size of the memory
    'capacity': 150000,

    # torch device to use for the model
    'device': 'cpu',
    # torch optimizer to be used
    'optim': 'adam',
    'lr': 3e-4,
}

DQNConfig = {
    'name': 'dqn',

    # parameter for eps-greedy policy
    'eps_start': .5,
    'eps_end': .05,
    'eps_decay': 4000,

    # discount factor
    'gamma': .99,

    # update method ('periodic' or 'soft')
    'update_method': 'soft',
    # update frequence for periodic, ratio for soft
    'target_update': 1000,
    'tau': .995,
}
DQNConfig = DQNConfig | GlobalConfig


DDPGConfig = {
    'name': 'ddpg',

    # parameter for eps-greedy policy
    'std_start': .5,
    'std_end': .05,
    'std_decay': 4000,

    # discount factor
    'gamma': .99,

    # parameter for soft update
    'tau': .995,
}
DDPGConfig = DDPGConfig | GlobalConfig


TD3Config = {
    'name': 'td3',

    # parameter for eps-greedy policy
    'std_start': .5,
    'std_end': .05,
    'std_decay': 4000,

    # discount factor
    'gamma': .99,

    # parameter for soft update
    'tau': .995,

    'policy_delay': 2,

    # target smoothing
    'target_std': .2,
    'target_clipping': .5,
}
TD3Config = TD3Config | GlobalConfig


SACConfig = {
    'name': 'sac',

    # discount factor
    'gamma': .99,

    # parameter for soft update
    'tau': .995,

    # entropy weight
    'alpha': .1,

    # std clamping
    'max_std': 2,
    'min_std': -20,

    # temperature tuning
    'autotune': True,
    'alpha_lr': 3e-4,

    # checkpoints
    'eval': True,
    'eval_rate': 20000
}
SACConfig = SACConfig | GlobalConfig
