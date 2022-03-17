GlobalConfig = {
    # width of the models' hidden layers
    'model_shape': [128],
    'n_steps': 1000000,
    'pre_run_steps': 100000,
    'greedy_steps': 980000,
    'batch_size': 64,

    # size of the memory
    'capacity': 150000,

    # torch device to use for the model
    'device': 'cuda',
    # torch optimizer to be used
    'optim': 'adam',
    'lr': 1e-3,
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
    'alpha': .2,

    # std clamping
    'max_std': 2,
    'min_std': -20,
}
SACConfig = SACConfig | GlobalConfig
