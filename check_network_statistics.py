"""
    Generate some networks statistics
"""

import networkx as nx
import pandas as pd
import numpy as np
import glob
import sys
import os

# staging
current_dir = os.getcwd()

network_dir = os.path.join(current_dir, 'networks')

def process_graph_statistics(graph):

    # Calculate the clustering
    clust = nx.clustering(graph)
    print(f'{clust} general clustering')
    print('---------------------------')
    # Calculate the number of clusters
    num_clusters = np.sum([clust[key] for key in clust.keys()])
    print(f'{num_clusters} total number of clusters')
    print('---------------------------')
    # Calcualte the average clustering of the graph
    avg_clustering = nx.average_clustering(graph)
    print(f'{avg_clustering} average clustering')
    print('---------------------------')
    # Calculate the number of triangles present
    triangls = nx.triangles(graph)
    print(f'{triangls} number of triangles')
    print('---------------------------')
    # Calculate the total number of triangles
    num_triangles = np.sum([triangls[key] for key in triangls.keys()])
    print(f'{num_triangles} total number of triangles')
    print('---------------------------')

# load the various files
files = glob.glob(network_dir+ '\*.npz')

# loop over the files
for ifile in files:
    num_triangles,  num_clusters = 0, 0
    matrix = np.load(ifile, allow_pickle=True)['matrix']

    df = pd.DataFrame(matrix)
    graph = nx.from_pandas_adjacency(df)
    process_graph_statistics(graph)


