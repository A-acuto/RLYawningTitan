"""
    Plot showing the deployment performances of the agents against the random agent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import re
import os

from RL_plottings import plot_deployment_performances

# staging
current_dir = os.getcwd()

figures = os.path.join(current_dir, 'figures')
data_dir = os.path.join(current_dir, 'results_data')

plt.style.use(os.path.join(figures, 'figure_style.mplstyle'))

# Set the network sizes
network_size = ['22', '55', '60']

# load the random agent
random_data = pd.read_csv(os.path.join(data_dir, 'random_agents_performances_deployment.csv'),
                          header=0)

# Plotting order
order = ['noscenario', 'compromise', 'isolate', 'mix', 'low_red', 'high_red']
order_label = ['noscenario', 'compromise', 'isolate', 'mix', 'low', 'high']

# instantiate a dictionary to collect the various scores
full_results = dict()

# loop over the network size
for isize in network_size:
    mean_train, mean_deploy = [], []
    std_train, std_deploy = [], []

    # grab the data
    training =os.path.join(data_dir, f'agents_performances_training_real_network_{isize}.csv')
    deployment = os.path.join(data_dir, f'agents_performances_deploy_real_network_{isize}.csv')

    traindata = pd.read_csv(training, names=['name', 'reward', 'lenght', 'mode'], header=0)
    deploydata = pd.read_csv(deployment, names=['name', 'reward', 'lenght', 'mode'], header=0)

    # clean the files
    traindata['reward'] = traindata['reward'].apply(lambda x: x[1:-1]).astype(float)
    deploydata['reward'] = deploydata['reward'].apply(lambda x: x[1:-1]).astype(float)

    # loop in the order labels
    for itype in order_label:
        mean_train.append(traindata.query(f'mode == "{itype}"')['reward'].mean())
        std_train.append(traindata.query(f'mode == "{itype}"')['reward'].std())
        mean_deploy.append(deploydata.query(f'mode == "{itype}"')['reward'].mean())
        std_deploy.append(deploydata.query(f'mode == "{itype}"')['reward'].std())

    # store the results
    full_results[isize] = {'mean_train': mean_train,
                            'mean_deploy': mean_deploy,
                            'std_train': std_train,
                            'std_deploy': std_deploy}

# create the x locations
points = np.linspace(1, 6, 6)
xlow = points - 0.2
xhigh = points + 0.2

# call the plotter function
plot_deployment_performances(order,
                             network_size,
                             random_data,
                             full_results,
                             xlow, xhigh, points)


