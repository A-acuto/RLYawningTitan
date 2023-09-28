# temp this will cover the various training performances of the various trained models

import numpy as np
import pandas as pd
import glob
import sys
import os

from RL_plottings import plot_training_performance

# Staging
current_dir = os.getcwd()

output_dir = os.path.join(current_dir, 'networks')
logs_dir = os.path.join(current_dir, 'logs_dir')
plot_dir = os.path.join(current_dir, 'figures')


# load the path models
models_names = ['PPO', 'A2C', 'DQN']
models_paths = [os.path.join(logs_dir, imodel) for imodel in models_names]
network_size = [18, 50, 100]
model_pars = ['NoHyperPars', 'df_075', 'lr_001']
# Plotting colors
colors = ['blue', 'green', 'orange']

# I should loop in a different way - No hyper pars, LR an DF, then size and loop over the various
for itype in model_pars:
    for isize in network_size:
        monitor_data = []
        for index_algo in range(len(models_paths)):
            if itype == model_pars[0]:
                lookUp_data = models_paths[index_algo] + \
                              f'\\{models_names[index_algo]}_{isize}_nodes.monitor.csv'
            else:
                lookUp_data = models_paths[index_algo] + \
                              f'\\{models_names[index_algo]}_{isize}_nodes_{itype}.monitor.csv'
            x = glob.glob(lookUp_data)[0]
            # append everything into a list
            monitor_data.append(pd.read_csv(x, skiprows=2, names=['Reward', 'Lenght', 'Time']))

        plot_training_performance(monitor_data, xlabel='Sampled timesteps',
                                  ylabel='Rewards',
                                  title=f"Models {itype} - {isize} nodes",
                                  plot_name=f'training_performances_{isize}_nodes_{itype}',
                                  model_names=models_names,
                                  colors=colors,
                                  save_plot= True)