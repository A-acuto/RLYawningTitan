"""
    Function to call and generate networks to re-use it across different tests and trainings.
"""

import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
import pickle

from yawning_titan.envs.generic.helpers import network_creator
from yawning_titan.config.game_config.game_mode_config import GameModeConfig
from yawning_titan.config.game_modes import default_game_mode_path, default_game_mode_tests_path
from yawning_titan.config.network_config.network_config import NetworkConfig


def dump(object, name: str):
    """
        Simple function to dump pickle files
    :param object:
    :param name:
    :return:
    """

    if not name.endswith('.pkl'):
        name = name +'.pkl'

    pickle.dump(object, open(name, 'wb'))

def load(name:str):
    """
        pickle loader
    :param name:
    :return:
    """

    if not name.endswith('.pkl'):
        name = name +'.pkl'

    return pickle.load(open(name, 'rb'))


def create_network(n_nodes: int, connectivity: float,
                   output_dir: str, filename: str,
                   save_matrix: bool = True,
                   save_graph: bool = False) -> [np.ndarray, dict]:

    """
        Function to create networks plus save it for reuse
    :param n_nodes:
    :return:
    """

    matrix, positions = network_creator.create_mesh(size=n_nodes, connectivity=connectivity)

    nodes = [str(i) for i in range(n_nodes)]

    df = pd.DataFrame(matrix, index=nodes, columns=nodes)

    graph = nx.from_pandas_adjacency(df)

    filen = output_dir + f'/' + filename

    if save_matrix:
        dump([matrix, positions], filen)
    if save_graph:
        dump(graph, filen)

    return (matrix, positions)

def main():

    outdir = f'./networks'

    create_network(50, 0.4, output_dir=outdir, filename='test', save=True)

#if __name__ == '__main__':
#    main()