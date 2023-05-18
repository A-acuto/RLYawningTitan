import logging
import numpy as np
import pandas as pd
import gym
from statistics import mean
import networkx as nx
import random
from stable_baselines3 import PPO  # agents
from stable_baselines3 import A2C, DQN  # agents

from stable_baselines3.ppo import MlpPolicy as PPOMlp  # policies
from stable_baselines3.a2c import MlpPolicy as A2CMlp # policies
from stable_baselines3.dqn import MlpPolicy as DQNMlp  # policies

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from tabulate import tabulate

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.helpers import network_creator
from yawning_titan.envs.generic.core.network_interface import NetworkInterface

from yawning_titan.config.game_config.game_mode_config import GameModeConfig
from yawning_titan.config.game_modes import default_game_mode_path, default_game_mode_tests_path
from yawning_titan.config.network_config.network_config import NetworkConfig


log_dir = './logs_dir/'

random_seeds = [2022, 14031879, 23061912, 6061944, 17031861]

#checkpoint_callback = CheckpointCallback(save_freq=10, save_path=log_dir, name_prefix='rl_model_check')

# setup the game mode from specific yaml file
game_mode = GameModeConfig.create_from_yaml(default_game_mode_tests_path())

# standard entry nodes
entry_nodes = ['3', '14', '10']

#   matrix, node_positions = network_creator.gnp_random_connected_graph(18,0.5)


matrix, positions = network_creator.create_18_node_network()

network = NetworkConfig.create_from_args(matrix=matrix, positions=positions, entry_nodes=entry_nodes)

# setup for the loops
dir_agent = ['PPO/', 'A2C/', 'DQN/']
agents_rl = [PPO, A2C, DQN]
policy_rl = [PPOMlp, A2CMlp, DQNMlp]
names = ['PPO_std', 'A2C_std', 'DQN_std', 'PPO_lr_001','A2C_lr_001', 'DQN_lr_001',
         'PPO_df_075','A2C_df_075', 'DQN_df_075']
timesteps = 5e5
count = 0

for i in range(len(names)):

    agent = agents_rl[count]
    policies = policy_rl[count]
    dirs = dir_agent[count]

    # here enters the random seed! - I must use them in the testing phase.
    network_interface = NetworkInterface(game_mode=game_mode, network=network)

    red = RedInterface(network_interface)
    blue = BlueInterface(network_interface)

    env = GenericNetworkEnv(
        red,
        blue,
        network_interface,
        print_metrics=True,
        show_metrics_every=50,
        collect_additional_per_ts_data=True,
        print_per_ts_data=False)

    check_env(env, warn=True)

    env.reset()
    env = Monitor(env, log_dir+'/'+str(dirs))

    if i > 2 and i <= 5:  # optimise learning rate
        chosen_agent = agent(policies, env, verbose=1, learning_rate = 0.01)
    if i > 5:  # optimise the discount factor
        chosen_agent = agent(policies, env, verbose=1, gamma = 0.75)
    else:  # standard case
        chosen_agent = agent(policies, env, verbose=1)

    x = chosen_agent.learn(total_timesteps=timesteps)
    print(x)
    sys.exit()
    #chosen_agent.save(log_dir+'/'+str(dirs) + names[i])

    count += 1
    if count > 2:
        count = 0