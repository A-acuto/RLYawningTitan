"""
    General object for calling and creating routines for plotting RL results and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import uniform_filter1d
import sys
import os
# Load the function to generate networks, or load pkl files
import generate_test_networks as gnt

# Staging
current_dir = os.getcwd()

output_dir = os.path.join(current_dir, 'networks')
logs_dir = os.path.join(current_dir, 'logs_dir')
plot_dir = os.path.join(current_dir, 'figures')

# Load the figure style, this is optional
plt.style.use(os.path.join(plot_dir, 'figure_style.mplstyle'))

def plot_training_performance(data, xlabel:str, ylabel:str):
    """
        Function to plot the performances from the monitoring of the environment
    :return:
    """


def plot_evaluations():
    """
        Function to plot the performances of the models and the random agent
    :return:
    """
    return ()


def main():

    # load the path models
    models_names = ['PPO', 'A2C', 'DQN']
    models_paths = [os.path.join(logs_dir, imodel) for imodel in models_names]
    network_size = [18, 50, 100]

    for isize in network_size:
        for index_algo in range(len(models_paths)):
            print(models_paths[index_algo] + f'\*{isize}.monitor.csv')
            x = glob.glob(models_paths[0] + f'\*{isize}*.monitor.csv')
            print(x)
    # process the monitors
    monitors = []


    sys.exit()

    files =['\PPO_std_100*','\A2C_std_100*', '\DQN_std_100*']

    print(log_dir + models[0])
    monitors = []
    data = []
    for i in range(3):
        monitors.append(glob.glob(log_dir + models[i] + files[i] + '.monitor.csv'))
        print(monitors[i])
        data.append(pd.read_csv(monitors[i][0], skiprows=2, names=['Reward', 'Lenght', 'Time']))
        print(data[i]['Time'][len(data[i])-1]/60.)
    print(monitors)

    colors= ['blue', 'green', 'orange']

    plt.figure(figsize=(10, 8))
    for i in range(3):
        timesteps = np.linspace(0, len(data[i]), len(data[i]))
        running_mean = uniform_filter1d(data[i]['Reward'], size= 5)
        temp_label = lbs[i] + ', Final value ' + str(np.ceil(running_mean[-2]))
        plt.plot(timesteps, data[i]['Reward'], alpha=0.5, color=colors[i], label=temp_label)
        plt.plot(timesteps, running_mean, color=colors[i],
                 linestyle='dashed')
        plt.xscale('log')
        plt.xlabel('Sampled timesteps')
        plt.ylabel('Rewards')
    plt.title('No hyperparameter tuning - 100 nodes')
    plt.legend(frameon=True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('plots/no_hyper_pars_tuning_100_nodes.png', dpi=150)
    plt.savefig('plots/no_hyper_pars_tuning_100_nodes.pdf', dpi=150)


if __name__ == '__main__':
    main()