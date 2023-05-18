import logging
import numpy as np
import pandas as pd
import gym
from statistics import mean
import networkx as nx
import random

from stable_baselines3 import PPO as PPO # agents

from stable_baselines3.ppo import MlpPolicy as policy # policies

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

import generate_test_networks as gtn
import glob


log_dir = './logs_dir/'

random_seeds = [2022, 14031879, 23061912, 6061944, 17031861]

game_mode = GameModeConfig.create_from_yaml(default_game_mode_tests_path())

network_dir = './networks/'

ntws = glob.glob(network_dir + 'common_network_50.pkl')
if len(ntws) == 1:
    matrix, positions = gtn.load(ntws[0])

else:
    matrix, positions = gtn.create_network(
        n_nodes = 50 , connectivity=0.4,
        output_dir = network_dir, filename = 'common_network_50',
        save_matrix = True,
        save_graph = False)

print(matrix, positions)
# standard entry nodes

entry_nodes = ['3', '14', '10', '45', '23', '30']

network = NetworkConfig.create_from_args(matrix=matrix, positions=positions, entry_nodes=entry_nodes)

# setup for the loops
dir_agent = 'PPO/' #, 'A2C/', 'DQN/']

names = ['ppo_std_test_size2']
timesteps = 3000 #5e5
count = 0

print('ok')