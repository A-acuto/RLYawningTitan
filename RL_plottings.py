"""
    General object for calling and creating routines for plotting RL results and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import uniform_filter1d
import sys

import generate_test_networks as gnt

plt.style.use('./plots/figure_style.mplstyle')

def plot_training_performance(data, xlabel:str, ylabel:str):
    """
        Function to plot the performances from the monitoring of the environment
    :return:
    """



def main():
    #create_network(50, 0.4, output_dir=outdir, filename='test', save=True)

    outdir = './networks'
    gnt.create_network(50, 0.4, output_dir='./networks', filename='test', save=False)

    data = gnt.load(outdir+'/' + 'test')

    print(data)

    sys.exit()


    models = ['\PPO', '\A2C', '\DQN']
    lbs = ['PPO', 'A2C', 'DQN']

    files =['\PPO_std_100*','\A2C_std_100*', '\DQN_std_100*']
    log_dir = r'.\logs_dir'

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