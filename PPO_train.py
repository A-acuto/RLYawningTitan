import logging
import numpy as np
import pandas as pd
import gym
from statistics import mean
import networkx as nx
import random
from stable_baselines3 import PPO as PPO # agents

from stable_baselines3.ppo import MlpPolicy as policy # policies
#from stable_baselines3.ppo import CnnPolicy as policy # policies
#from stable_baselines3.ppo import MultiInputPolicy as policy # policies

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from tabulate import tabulate

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.helpers import network_creator
from yawning_titan.envs.generic.core.network_interface import NetworkInterface

from yawning_titan.config.game_config.game_mode_config import GameModeConfig
from yawning_titan.config.game_modes import default_game_mode_path, default_game_mode_tests_path,\
    default_game_mode_low_actions_path
from yawning_titan.config.network_config.network_config import NetworkConfig

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

import generate_test_networks as gtn
import glob

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback


log_dir = './logs_dir/'

random_seeds = [2022, 14031879, 23061912, 6061944, 17031861]

game_mode = GameModeConfig.create_from_yaml(default_game_mode_low_actions_path())

network_dir = './networks/'

ntws = glob.glob(network_dir + 'nodes_55_training.pkl')
if len(ntws) == 1:
    matrix, positions = gtn.load(ntws[0])

else:
    matrix, positions = gtn.create_network(
        n_nodes = 50 , connectivity=0.4,
        output_dir = network_dir, filename = 'common_network_50',
        save_matrix = True,
        save_graph = False)

# standard entry nodes

entry_nodes = ['3', '14', '10', '45', '23', '30']
#entry_nodes = ['3', '14', '10']

#matrix, positions = network_creator.gnp_random_connected_graph(100, 0.3)

#matrix, positions = network_creator.create_18_node_network()

network = NetworkConfig.create_from_args(matrix=matrix, positions=positions, entry_nodes=entry_nodes)

# setup for the loops
dir_agent = 'PPO/' #, 'A2C/', 'DQN/']
#names = ['PPO_std_50_nodes_cnn', 'PPO_lr_001_50_nodes_cnn', 'PPO_df_075_50_nodes_cnn']
#names = ['PPO_std_18_nodes_cnn', 'PPO_lr_001_18_nodes_cnn', 'PPO_df_075_18_nodes_cnn']
#names = ['PPO_std_50_nodes_ml', 'PPO_lr_001_50_nodes_ml', 'PPO_df_075_50_nodes_ml']
#names = ['PPO_std_18_nodes_ml', 'PPO_lr_001_18_nodes_ml', 'PPO_df_075_18_nodes_ml']

names = ['ppo_55_low_actions']
timesteps = 5e5
count = 0

for i in range(len(names)):

    agent = PPO #agents_rl[count]
    policies = policy #policy_rl[count]

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
    env = Monitor(env, log_dir+'/'+str(dir_agent)+ names[i])
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback = EvalCallback(env, best_model_save_path=log_dir+'/'+str(dir_agent) + names[i],
                                 eval_freq=1000,log_path=log_dir+'/'+str(dir_agent) + names[i],
                                 callback_after_eval=stop_train_callback, deterministic=False, verbose=1)


    if i==0:  # optimise learning rate
        print(i)
        chosen_agent = agent(policies, env, verbose=1, n_steps=1024, n_epochs=50, gamma=0.75)
    if i==2:  # optimise the discount factor
        chosen_agent = agent(policies, env, verbose=1, gamma = 0.75, learning_rate = 0.1)
    else:  # standard case
        chosen_agent = agent(policies, env, verbose=1)

#    eval_callback = EvalCallback(env, best_model_save_path=log_dir+'/'+str(dir_agent) + names[i],
#                                 log_path=log_dir+'/'+str(dir_agent) + names[i], eval_freq=512,
#                                deterministic=True, render=False)

#    new_logger = configure(log_dir+'/'+str(dir_agent) + names[i], ["stdout", "json"])
#    chosen_agent.set_logger(new_logger)
    x = chosen_agent.learn(total_timesteps=timesteps, callback=eval_callback)
    chosen_agent.save(log_dir+'/'+str(dir_agent) + names[i])

