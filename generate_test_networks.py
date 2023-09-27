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

import os
current_dir= os.getcwd()

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
    :param
    n_nodes: number of nodes of the network
    connectivity: percentage of edges connecting the nodes
    output_dir: where to place the networks
    filename: save the networks
    save_matrix: save in a npz/pkl objects the matrix and the edges
    save_graph: save the nx output

    :return:
    matrix: np.array containing the matrix describing the graph
    positions: node positions
    """

    # Use the yawning titan generator to create the mesh of given size
    matrix, positions = network_creator.create_mesh(size=n_nodes, connectivity=connectivity)

    nodes = [str(i) for i in range(n_nodes)]

    # create the dataframe for the graph
    graph_df = pd.DataFrame(matrix, index=nodes, columns=nodes)

    # create a graph for visualisation
    graph = nx.from_pandas_adjacency(graph_df)

    # check if the filename has the right ending or not
    if filename.endswith('.npz'):
        filen = os.path.join(output_dir, filename)
    else:
        filen = os.path.join(output_dir, filename+'.npz')

    # prefer the npz for easier sharing instead of pkl files.
    if save_matrix:
        np.savez(filen, matrix=matrix, connections=positions)
#        dump([matrix, positions], filen)
    if save_graph:
#        dump(graph, filen)
        np.savez(filen, graph=graph)

    return (matrix, positions)

def main():

    # example running
    outdir = os.path.join(current_dir, 'networks')

    # example nodes and connectivity
    n_nodes = 50
    connectivity = 0.4

    # showing the example
    matrix, _ = create_network(n_nodes, connectivity, output_dir=outdir,
                               filename='test', save_matrix=False)
    # the positions are not relevant in this specific example

    nodes = [str(i) for i in range(n_nodes)]

    graph_df = pd.DataFrame(matrix, index=nodes, columns=nodes)

    # generate the graph using the adjacency matrix
    graph = nx.from_pandas_adjacency(graph_df)

    # seed the position for replicability
    my_pos = nx.spring_layout(graph, seed=99)

    plt.figure(figsize=(9.5, 6.5), dpi=150)
    nx.draw(graph, with_labels=True,
            node_size=450, node_shape='8', pos=my_pos,
            verticalalignment='center', horizontalalignment='left', clip_on=False,
            font_weight='normal', linewidths=0.5, alpha=1, width=0.8)
    plt.title('Example graph')
    plt.axis('off')
    # plt.tight_layout() # sometimes it complains
    plt.show()

if __name__ == '__main__':
    main()



