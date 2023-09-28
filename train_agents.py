"""
    This code is an example for setting up the training of the various agents using the algorithms
    present and using the current version of YAWNING TITAN.
"""

import logging
import numpy as np
import pandas as pd
import gym
import networkx as nx
import random

# load the agents
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DQN

# load the policies
from stable_baselines3.a2c import MlpPolicy as A2C_policy
from stable_baselines3.ppo import MlpPolicy as PPO_policy
from stable_baselines3.dqn import MlpPolicy as DQN_policy

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

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

import generate_test_networks as gtn
import glob
import os

# random seeds
random_seeds = [2022, 14031879, 23061912, 6061944, 17031861]

# get the current directory
current_dir = os.getcwd()

# directories
log_dir = os.path.join(current_dir, 'logs_dir')
network_dir = os.path.join(current_dir, 'networks')

# Specify algorithms, policies and saving directories
algorithms = ['PPO', 'A2C', 'DQN']
agents = [PPO, A2C, DQN]
policies = [PPO_policy, A2C_policy, DQN_policy]
dir_agent = [os.path.join(log_dir, imodel) for imodel in algorithms]
timesteps = 5e5 # check with lower timesteps
model_names = ['PPO_std', 'A2C_std', 'DQN_std']  # for this example we can show only the stand setup,

# get the network network nodes
standard_example = [18, 50, 100]  # if you want to test the "unseen" networks use 22/55/60

# entry nodes
network_entry = [['3', '5', '10'],  # 18
           ['3', '10', '15', '25', '34', '45', '7'],  # 50
           ['4', '10', '20', '30', '40', '55', '76', '78', '12', '88', '90']]  # 100

# loop over the network size
for index, isize in enumerate(standard_example):
    network_load = glob.glob(os.path.join(network_dir, f'synthetic_{isize}*.npz'))

    if len(network_load) == 1:
        network_files = np.load(network_load[0], allow_pickle=True)
        matrix = network_files['matrix']
        positions = dict(np.ndenumerate(network_files['connections']))[()]  # convert the positions nd array to dict
    else:
        if isize == 18:
            matrix, positions = network_creator.create_18_node_network()
        else:
            matrix, positions = gtn.create_network(
                n_nodes=isize,
                connectivity=0.6,  # standard connectivity
                output_dir=network_dir,
                filename=f'synthetic_{isize}',
                save_matrix=True,
                save_graph=False)

    # need to load the various networks
    network = NetworkConfig.create_from_args(matrix=matrix, positions=positions, entry_nodes=network_entry[index])

    # Loop over the algorithms
    for ialgorithm in range(len(algorithms)):

        agent = agents[ialgorithm]
        policies = policies[ialgorithm]
        model_dir = os.path.join(dir_agent[ialgorithm], algorithms[ialgorithm])
        model_name = os.path.join(model_dir, model_names[i]+f'_{isize}')

        print(f'Starting the agent using {algorithms[ialgorithm]} algorithm')

        # here enters the random seed! - I must use them in the testing phase.
        network_interface = NetworkInterface(game_mode=game_mode, network=network)

        # generate the red and blue agents
        red = RedInterface(network_interface)
        blue = BlueInterface(network_interface)

        # generate the network environment
        env = GenericNetworkEnv(
            red,
            blue,
            network_interface,
            print_metrics=True,
            show_metrics_every=50,
            collect_additional_per_ts_data=True,
            print_per_ts_data=False)

        # check the network
        check_env(env, warn=True)

        # reset the environment
        env.reset()

        # setup the monitor to check the training
        env = Monitor(env, model_name)

        # define callback to stop the training
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
        eval_callback = EvalCallback(env,
                                     best_model_save_path=model_dir,  # save the model
                                     eval_freq=1000,
                                     log_path=model_dir, # save the logs
                                     callback_after_eval=stop_train_callback,
                                     deterministic=False,
                                     verbose=1)

        # instantiate the agent - here we can set the various hyper parameters as the
        # Learning rate - tested to  learning_rate = 0.01 and the gamma = 0.75
        if isize == 100 and algorithms[ialgorithm] =='DQN': # adapt in case of buffer size
            chosen_agent = agent(policies, env, verbose=1, buffer_size=10000, normalize_advantage=True)
        else:
            chosen_agent = agent(policies, env, verbose=1, normalize_advantage=True)

        # Train the agent
        _ = chosen_agent.learn(total_timesteps=timesteps, callback=eval_callback)
        # save the trained-converged model
        chosen_agent.save(model_name)



