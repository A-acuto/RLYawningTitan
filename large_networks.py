import logging
import numpy as np
import pandas as pd
import gym
from statistics import mean
import networkx as nx
import random
from stable_baselines3 import PPO  # agents

from stable_baselines3.ppo import MlpPolicy as PPOMlp  # policies

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

import psutil
import sys
import platform

log_dir = './logs_dir/'

random_seeds = [2022, 14031879, 23061912, 6061944, 17031861]

# test on the size of the network
game_mode = GameModeConfig.create_from_yaml(default_game_mode_tests_path())

matrix, positions = network_creator.create_mesh(size=5000)

print(platform.machine(),  ' machine')
print(platform.version(),  ' version')
print(platform.platform(),  ' platform')
print(platform.processor(),  ' processor')
print(psutil.cpu_percent(), ' CPU percent')
print(psutil.virtual_memory(), ' virtual_memory')
print(psutil.virtual_memory().available*100/psutil.virtual_memory().total, ' virtual memory used')


# print(matrix)
network = NetworkConfig.create_from_args(matrix=matrix, positions=positions)
network_interface = NetworkInterface(game_mode=game_mode, network=network)
print('network created')
print(platform.machine(),  ' machine')
print(platform.version(),  ' version')
print(platform.platform(),  ' platform')
print(platform.processor(),  ' processor')
print(psutil.cpu_percent(), ' CPU percent')
print(psutil.virtual_memory(), ' virtual_memory')
print(psutil.virtual_memory().available*100/psutil.virtual_memory().total, ' virtual memory used')

red = RedInterface(network_interface)
blue = BlueInterface(network_interface)
print('agents done')
print(platform.machine(),  ' machine')
print(platform.version(),  ' version')
print(platform.platform(),  ' platform')
print(platform.processor(),  ' processor')
print(psutil.cpu_percent(), ' CPU percent')
print(psutil.virtual_memory(), ' virtual_memory')
print(psutil.virtual_memory().available*100/psutil.virtual_memory().total, ' virtual memory used')
env = GenericNetworkEnv(
    red,
    blue,
    network_interface,
    print_metrics=True,
    show_metrics_every=50,
    collect_additional_per_ts_data=True,
    print_per_ts_data=False)

check_env(env, warn=True)
print('check done')
print(platform.machine(),  ' machine')
print(platform.version(),  ' version')
print(platform.platform(),  ' platform')
print(platform.processor(),  ' processor')
print(psutil.cpu_percent(), ' CPU percent')
print(psutil.virtual_memory(), ' virtual_memory')
print(psutil.virtual_memory().available*100/psutil.virtual_memory().total, ' virtual memory used')

chosen_agent = PPO(PPOMlp, env, verbose=1)

x = chosen_agent.learn(total_timesteps=1000)

sys.exit()

