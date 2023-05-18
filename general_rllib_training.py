"""
    So in this code I will set up a general rllib training code so I can build and train various
    agents using different rl sistems in a smarter way
    Basically I want to pass some information to this software via command line and then use them properly
"""

import logging
import numpy as np
import pandas as pd
import gym
from statistics import mean
import networkx as nx
import random

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

# ray stuff
from ray.tune.registry import register_env

from ray.air.config import RunConfig
from ray.air import Checkpoint
from ray.train.rl.rl_predictor import RLPredictor
from ray.train.rl.rl_trainer import RLTrainer
from ray.air.result import Result
from ray.air.config import ScalingConfig
from ray.tune.tuner import Tuner
import ray.air as air
import ray

game_mode = GameModeConfig.create_from_yaml(default_game_mode_tests_path())

network_dir = './networks/'
save_dir = "saved_models_rllib"

ntws = glob.glob(network_dir + 'nodes_22_training.pkl')
if len(ntws) == 1:
    matrix, positions = gtn.load(ntws[0])

else:
    matrix, positions = gtn.create_network(
        n_nodes=50, connectivity=0.4,
        output_dir=network_dir, filename='common_network_50',
        save_matrix=True,
        save_graph=False)

entry_nodes = ['3', '14', '10', '45', '23', '30']

network = NetworkConfig.create_from_args(matrix=matrix, positions=positions, entry_nodes=entry_nodes)

network_interface = NetworkInterface(game_mode=game_mode, network=network)

red = RedInterface(network_interface)
blue = BlueInterface(network_interface)
print('network initiated')
env = GenericNetworkEnv(
    red,
    blue,
    network_interface,
    print_metrics=True,
    show_metrics_every=50,
    collect_additional_per_ts_data=True,
    print_per_ts_data=False)


def env_creator(env_config={}):
    return YT()


# env_creator = lambda config: env
# register_env("my_env", env_creator)

train_steps = 5e5
#learning_rate = 1e-3
save_dir = "saved_models_rllib_new"
env_name = 'my_env'

algos = ['APPO'] #, 'A2C', 'APPO', 'PG', 'SAC', 'QMIX']

ray.init(local_mode=True, num_cpus=1,)

from ray.rllib.algorithms.appo import APPOConfig

for i in algos:
    print('i')
    env.reset()
    env_creator = lambda config: env
    register_env('my_env', env_creator)

    # training and saving
    trained_model = i + '_22_nodes_exp'

    config = APPOConfig().training(grad_clip=30.0)
#    config = config.resources(num_cpus_per_worker=1)
#    config = config.rollouts(num_rollout_workers=16)
    config =  config.environment(env = 'my_env')
#    algo = config.build()

    analysis = Tuner(
        "APPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": train_steps, 'episode_reward_mean': -50},
            local_dir=save_dir,
            name=trained_model,
            #scaling_config=ScalingConfig(num_workers=8, use_gpu=False, trainer_resources={"CPU": 1}),
            checkpoint_config=air.CheckpointConfig(
                num_to_keep =5,
                checkpoint_at_end=True,
            ),
        ),
        param_space=config.to_dict(),
    ).fit()
