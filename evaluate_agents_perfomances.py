"""
    Code to deploy and test the various trained algorithms
"""

import numpy as np
import pandas as pd
import glob
import sys
import os

# load the various algorithms
from stable_baselines3 import PPO  # agents
from stable_baselines3 import A2C  # agents
from stable_baselines3 import DQN  # agents

# load the random agent
from yawning_titan.agents.random import RandomAgent

from yawning_titan.config.game_config.game_mode_config import GameModeConfig
# Load the various game modes  - standard, low red skills, high red skills and rnd seeded files
from yawning_titan.config.game_modes import default_game_mode_path, default_game_mode_tests_path, \
    default_game_mode_tests_rnd_path, default_game_mode_low_red_skills_rnd_path, \
    default_game_mode_high_red_skills_rnd_path
from yawning_titan.config.network_config.network_config import NetworkConfig

from stable_baselines3.common.monitor import Monitor

# general yawning titan components
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.helpers import network_creator
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.core.network_interface_explore import NetworkInterfaceExplore

# stable baselines evaluator
from stable_baselines3.common.evaluation import evaluate_policy

import generate_test_networks as gtn

current_dir = os.getcwd()
network_dir = os.path.join(current_dir, 'networks')
results_dir = os.path.join(current_dir, 'results_data')
model_dir = os.path.join(current_dir, 'logs_dir')


# these are useful for the comparison, also worth modifying one to spread the random seed inside
# these seeds are hard-coded to the existing yaml files.
random_seeds = [2022, 14031879, 23061912, 6061944, 17031861]


# network entries
network_entry = [['3', '5', '10'],  # 18
           ['3', '10', '15', '25', '34', '45', '7'],  # 50
           ['4', '10', '20', '30', '40', '55', '76', '78', '12', '88', '90']]  # 100

# network size
network_sizes = [18, 50, 100]

# Naming and algorithms
agents_algos = ['PPO', 'A2C', 'DQN']
agents = [PPO, A2C, DQN]

# Gather relevant data, it will take some time
model_names = []
rewards = []
episode_lenght = []

# loop on the network sizes
for index, net_size in enumerate(network_sizes):

    network_image = glob.glob(os.path.join(network_dir, f'synthetic_{net_size}_nodes_network.npz'))[0]

    network_files = np.load(network_image, allow_pickle= True)
    matrix = network_files['matrix']
    positions = dict(np.ndenumerate(network_files['connections']))[()] # convert the positions nd array to dict

    # generate the network
    network = NetworkConfig.create_from_args(matrix=matrix, positions=positions,
                                             entry_nodes = network_entry[index])

    # now loop on the various random seeds
    for iseed in random_seeds:
        # seeded testing - if you want to use one of the red agents changes use
        # default_game_mode_low_red_skills_rnd_path or default_game_mode_high_red_skills_rnd_path
        game_mode = GameModeConfig.create_from_yaml(default_game_mode_tests_rnd_path(iseed))

        # Now test the agent on specific scenarios:
        # standard : standard yawning titan scenario initiated, if this mode num_extension is ignored
        # compromise : some nodes (num_extensions) are randomly infected
        # isolate : some nodes (num_extensions) are randomly isolated
        # mix : some nodes (num_extensions) are isolated and some are compromised

        network_interface = NetworkInterfaceExplore(game_mode=game_mode,
                                                    network=network,
                                                    num_extension=3,  # number of nodes impacted by the extension
                                                    extension='standard')

        # load the red and blue agent
        red = RedInterface(network_interface)
        blue = BlueInterface(network_interface)

        # initialise the environment
        env = GenericNetworkEnv(red, blue, network_interface, print_metrics=False)

        # reset the environment
        env.reset()

        # Now we need to loop over the models - trained agents
        for idx, iagent in enumerate(agents):
            # collect all the possible models trained for each algorithm at given network size
            trained_models = glob.glob(os.path.join(model_dir, agents_algos[idx]) +
                                       f'\\{agents_algos[idx]}_{net_size}_*.zip' )

            # instantiate the agent
            algorithm = agents[idx]
            # loop over the models trained
            for imodel in trained_models:
                print(agents_algos[idx], net_size, imodel)
                model_names.append(imodel)
                agent = algorithm.load(imodel)

                # evaluate the policy
                eval_pol = evaluate_policy(agent,
                                           Monitor(env),
                                           return_episode_rewards=True,
                                           deterministic=False,
                                           n_eval_episodes=1)
                rewards.append(eval_pol[0])
                episode_length.append(eval_pol[1])

                # delete the agent loaded with the trained agent and reset the environment
                del agent
                env.reset()

        # now test the random agent
        R_agent = RandomAgent(env.action_space)
        model_names.append(f'random_agent_{net_size}')
        # reset the observations and rewards
        obs = env.reset()
        reward = 0

        # final reward
        rw = 0
        for iaction in range(500):  ## 500 timesteps
            action = Ragent.predict(obs, reward, '')
            obs, reward, done, info = env.step(action)

            rw += reward
            if done:
                env.reset()
#                print(rw, iaction, done)
                break
        env.reset()
        rewards.append(rw)
        episode_lenght.append(iaction)

# now dump all the relevant data into a dataframe containing all the models involved, the
# model names, rewards and the episode lenghts
summary_data = pd.DataFrame(data={'model': model_names,
                                  'reward': rewards,
                                  'lenght': episode_length})

# dump all the data into a csv
summary_data.to_csv(results_dir + 'agents_evaluation.csv', index=False)