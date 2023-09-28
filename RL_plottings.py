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

def plot_training_performance(data, xlabel:str, ylabel:str,
                              title: str,
                              model_names: list,
                              plot_name: str,
                              colors: list= [],
                              plot_mean:bool = True,
                              save_plot:bool = False):
    """
        Function to plot the performances from the monitoring of the environment
    :return:
    """

    # check if the list is not empty
    if not colors:
        colors= ['blue', 'green', 'orange']

    if isinstance(data, list):

        plt.figure(figsize=(10, 8))
        # loop over the models
        for i in range(len(data)):
            # create a series of timesteps
            timesteps = np.linspace(0, len(data[i]), len(data[i]))
            # smooth the training
            running_mean = uniform_filter1d(data[i]['Reward'], size=5)
            temp_label = model_names[i] + ', Final value ' + str(np.ceil(running_mean[-2]))
            plt.plot(timesteps, data[i]['Reward'],
                     alpha=0.5, color=colors[i], label=temp_label)
            if plot_mean:
                plt.plot(timesteps, running_mean, color=colors[i],
                         linestyle='dashed')

            plt.xscale('log')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(frameon=True)
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(plot_dir, plot_name + '.png'), dpi=200)
            plt.savefig(os.path.join(plot_dir, plot_name + '.pdf'), dpi=200)
        else:
            plt.show()
    else:
        raise Exception('Provide a list object')



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
                                      plot_name= f'training_performances_{isize}_nodes_{itype}',
                                      model_names=models_names,
                                      colors=colors)

        # sys.exit()
        #
        #
        # plt.figure(figsize=(10, 8))
        # for i in range(len(monitor_data)):
        #     timesteps = np.linspace(0, len(monitor_data[i]), len(monitor_data[i]))
        #     # smooth the training
        #     running_mean = uniform_filter1d(monitor_data[i]['Reward'], size=5)
        #     temp_label = models_names[i] + ', Final value ' + str(np.ceil(running_mean[-2]))
        #     plt.plot(timesteps, monitor_data[i]['Reward'],
        #              alpha=0.5, color=colors[i], label=temp_label)
        #     plt.plot(timesteps, running_mean, color=colors[i],
        #              linestyle='dashed')
        #     plt.xscale('log')
        #     plt.xlabel('Sampled timesteps')
        #     plt.ylabel('Rewards')
        # plt.title(f'No hyperparameter tuning - {isize} nodes')
        # plt.legend(frameon=True)
        # plt.tight_layout()
        # plt.show()
        # sys.exit()
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