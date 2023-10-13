"""
    General object for calling and creating routines for plotting RL results and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from scipy.ndimage import uniform_filter1d

import glob
import re
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
        Function to plot the performances from the Stable baselines monitors

        This function creates figures of the mean rewards over timesteps
        calculated from the environment monitor provided by Stable Baselines

    :parameter:
        data : list containing the dataframes composing the rewards over timesteps
        xlabel : xlabel of the plot
        ylabel : ylabel of the plot
        title : title of the plot
        model_names : names of the algorithms, used for titles and objects in the plot
        plot_name : final name of the figure
        colors : list of colors
        plot_mean : toggle to add the running mean of the measurements to the plot
        save_plot : toggle to show or save the figure.
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



def plot_evaluation_performance(sum_stats,
                                xlocs_ticks: np.array,
                                xlow: list,
                                xmid: list,
                                xhigh: list,
                                random_y: list,
                                random_ystd: list,
                                title_string: str):
    """
        This function allows to create figure containing an algorithm comparison
        while changing the hyper-parameters and the network size.

    :paramm:

    xlow: X positions of the datapoints for each algorithm with no hyper pars changes
    xmid: X positions of the datapoints for each algorithm with discount factor =0.75
    xhigh: X positions of the datapoints for each algorithm with learning rate modified
    xlocs_ticks : np.arry containing the X-ticks of the xaxis for the plots
    random_y : random agent scores (mean)
    random_ystd : random agent score standard deviations
    title_string : string containing the title of the plot

    :return:
    figure
    """

    # Generate some starting data for the plot
    colors, xaxis = [], []
    nohyp, disc_factor, lear_rate = [], [], []
    sigma_nohyp, sigma_disc_facto, sigma_learn_rate = [], [], []

    # box options for network size
    props = dict(boxstyle='square', facecolor='silver', alpha=0.7)

    # Loop over the summary statistics generate
    # this allows to summarise the statistics and
    # make the plotting easier
    for network_keys, results in sum_stats.items():
        for model, model_bit in results.items():
            xaxis.append(model)
            for sub, subpar in model_bit.items():
                if sub == 'std':
                    nohyp.append(subpar['rw_mean'])
                    sigma_nohyp.append(subpar['rw_std'])
                elif sub == 'df075':
                    disc_factor.append(subpar['rw_mean'])
                    sigma_disc_facto.append(subpar['rw_std'])
                elif sub == 'lr001':
                    lear_rate.append(subpar['rw_mean'])
                    sigma_learn_rate.append(subpar['rw_std'])

                if model == 'PPO':
                    colors.append('blue')
                elif model =='A2C':
                    colors.append('green')
                elif model == 'DQN':
                    colors.append('orange')

    # plot the datapoints
    plt.errorbar(xlow, nohyp, yerr=sigma_nohyp, fmt='D', label=r'$\gamma$=0.99, $\mathrm{LR}=3\times10^{-4}$')
    plt.errorbar(xmid, disc_factor, yerr=sigma_disc_facto, fmt='o', label=r'$\gamma$=0.75')
    plt.errorbar(xhigh, lear_rate, yerr=sigma_learn_rate, fmt='*', label=r'$\mathrm{LR}$=0.001')

    # add the random agent performances
    # specify the width
    xrandom=[np.linspace(-0.5,8.5,8), np.linspace(8.5,17.5,8),np.linspace(17.5, 26.5, 8)]

    # loop over the existing numbers
    for idxx, ix in enumerate(xrandom):
        # plot the random agent performances
        plt.fill_between(ix,
                         np.repeat(random_y[idxx], 8)-random_ystd[idxx],
                         np.repeat(random_y[idxx], 8)+random_ystd[idxx],
                         color='grey', alpha=0.5)
        # Add the labels
        if idxx == 2:
            plt.hlines(y=random_y[idxx], xmin=ix[0], xmax=ix[-1],
                       linestyles='dashed', color='k',
                       label='Random agent')
        else:
            plt.hlines(y=random_y[idxx], xmin=ix[0], xmax=ix[-1],
                       linestyles='dashed', color='k')
    # plot the vertical lines
    plt.vlines([8.5, 17.5], ymin=-7000, ymax=500,
               color='k', linestyles='dashed')

    # Add the text describing the network size
    plt.text(x=2, y=250, s='18 nodes', bbox=props)
    plt.text(x=11, y=250, s='50 nodes', bbox=props)
    plt.text(x=20, y=250, s='100 nodes', bbox=props)
    # plot the mean rewards
    plt.ylabel('Mean Rewards', fontweight='bold')
    # add the title
    plt.title(title_string, fontweight='bold')
    # location of the ticks for the names of the algorithms
    plt.xticks(xlocs_ticks, xaxis)
    plt.ylim(-5500, 500)
    plt.xlim(-0.1, 26.5)
    plt.legend(loc='lower left', title='Legend', frameon=True, framealpha=0.9)

def plot_multi_extension_comparison_size(network_size: list,
                                         plot_order: list,
                                         algorithm_order: list,
                                         order_label: list,
                                         agents_summary: dict,
                                         agent_random: pd.DataFrame,
                                         xlow: list,
                                         xmid: list,
                                         xhigh: list,
                                         ):
    """
        Routine to create the massive extension comparison using the
        evaluations obtained by each algorithms in the same networks as the
        training.
    :params:
        network_sizes : networks size to loop over
        plot_order : on which order we want to plot the data
        algorith_order : on which hyper-parameters loop
        order_label : list of labels of the x-ticks
        agents_summary : a dictionary containing all the summary statistics of the
                        various evalutations run
        agent_random  : random agent data collected
        xlow: X positions of the datapoints for each algorithm with no hyper pars changes
        xmid: X positions of the datapoints for each algorithm with discount factor =0.75
        xhigh: X positions of the datapoints for each algorithm with learning rate modified
    :return:
        figure
    """

    # create box for notes
    props = dict(boxstyle='square', facecolor='silver', alpha=0.7)

    # Create patches for the legend
    ppo = mpatches.Patch(color='blue', label='PPO')
    a2c = mpatches.Patch(color='orange', label='A2C')
    dqn = mpatches.Patch(color='darkgreen', label='DQN')
    rnd = mpatches.Patch(color='grey', label='RND')

    # create an unique panel
    gs = gridspec.GridSpec(nrows=3, ncols=1, width_ratios=[5], height_ratios=[5, 5, 5])

    # Loop over the size, each one has one horizontal panel
    for iplot,isize in enumerate(network_size):
        # select the random data that have the right size
        data_random = agent_random.query(f'size == {isize}')
        # create empty lists for values
        random_means, random_std = [], []

        # create an empty lists for the colors
        colors = []

        # create a series of lists for the values according
        # to the low, mid and high value for each X-points,
        # as well as the standard deviations
        vml, vmm, vmh = [], [], []
        sml, smm, smh = [], [], []

        # instantiate the subplot
        ax = plt.subplot(gs[iplot])
        # this loops vertically from all the entries valid - standard to skilled red
        for yvertical in plot_order:
            # select which data consider
            algorithms = agents_summary[yvertical][isize]
            # add the random data
            random_means.append(data_random.query(f'mod == "{yvertical}"')['mean'])
            random_std.append(data_random.query(f'mod == "{yvertical}"')['stddev'])

            # Loop over the algorithms
            for algorithm in algorithms.keys():
                # now loop on the various algorithms
                models = algorithms[algorithm]
                if algorithm == 'PPO':
                    pcolor='blue'
                elif algorithm == 'A2C':
                    pcolor='orange'
                elif algorithm == 'DQN':
                    pcolor='darkgreen'

                colors.append(pcolor)

                for indexx, itype in enumerate(algorithm_order):

                    if indexx == 0:
                        vml.append(models[itype]['rw_mean'])
                        sml.append(models[itype]['rw_std'])
                    elif indexx ==1:
                        vmm.append(models[itype]['rw_mean'])
                        smm.append(models[itype]['rw_std'])
                    elif indexx ==2:
                        vmh.append(models[itype]['rw_mean'])
                        smh.append(models[itype]['rw_std'])

         # finally plot the data
        plt.scatter(xlow, vml, color=colors, marker='D', alpha=0.9)
        plt.scatter(xmid, vmm, color=colors, marker='X', alpha=0.9)
        plt.scatter(xhigh,vmh, color=colors, marker='v', alpha=0.9)


        # now add the random agent values
        count = 0
        xticks_values = []
        for irandom in range(len(random_means)):

            if irandom == 0:
                low_tag= irandom
            high_tag=low_tag+2

            # dividers
            plt.vlines(xhigh[high_tag] + 0.5, ymin=-7000,
                       ymax=500, color='k', linestyles='dashed', alpha=0.7)

            # random agent
            plt.fill_between(xmid[low_tag:high_tag+1],
                             np.repeat(random_means[count], 3) - random_std[count],
                             np.repeat(random_means[count], 3) + random_std[count],
                             color='grey', alpha=0.5)
            plt.hlines(y=random_means[count], xmin=xmid[low_tag],
                       xmax=xmid[high_tag], linestyles='dashed', color='k',
                       label='Random agent')
            xticks_values.append(xmid[low_tag + 1])
            low_tag = high_tag+1
            count += 1

        # Plotting adjustments like legend, ticks or not
        if iplot == 0:
            plt.ylim(-3500, 250)
            plt.text(x=67, y=-2800, s=r'n=18', bbox=props)
            plt.legend(handles=[ppo, a2c, dqn, rnd], ncol=2, fontsize=8)
        elif iplot ==1 :
            plt.ylim(-5000, -100)
            plt.text(x=67, y=-4200, s=r'n=50', bbox=props)
        elif iplot ==2:
            plt.ylim(-6000, 30)
            plt.text(x=65, y=-5100, s=r'n=100', bbox=props)
        if iplot == 2:
            plt.xticks(xticks_values, order_label, rotation=30)
        else:
            ax.tick_params(labelbottom=False, direction='in', which='both')

        if iplot ==1:
            plt.ylabel('Mean Rewards', fontweight='bold')


def plot_deployment_performances(order: list,
                                 sizes: list,
                                 random_performances: pd.DataFrame,
                                 full_data: dict,
                                 xlow: list,
                                 xhigh: list,
                                 xpoints: list):
    """
        Routine to create the deployment comparison plot using the
        scores obtained by each algorithms in the training and realistic networks.

    :params:
        order : list of labels of the extension tested (fixed for all tabs)
        sizes : list of network sizes (22, 55, 60)
        random_performances : mean scores and standard deviations obtained by the Random agent
        full_data: dictionary containing the various mean and std dev of the scores of each algorithm
                   sub-divided by keys in network sizes and in the order specified before. In this
                   dictionary is present both the training and deployment results
        xlow: X positions of the datapoints for each algorithm for the training results
        xhigh X positions of the datapoints for each algorithm for the deploy results
        xpoints: center positions for the random scores
    :return:
        figure
    """

    # create a grid
    gs = gridspec.GridSpec(nrows=3, ncols=1, width_ratios=[5], height_ratios=[5, 5, 5])

    # create a label placing
    props = dict(boxstyle='square', facecolor='silver', alpha=0.7)

    # Loop over the sizes of network
    for iplot, isize in enumerate(sizes):
        ax = plt.subplot(gs[iplot])
        # select only the data from the specific network size
        data_by_size = full_data[isize]

        # select the random scores
        random_results = random_performances.query(f'size=={isize}')

        # in each ax object we adjust both the ticks, the text and legend items
        if isize == '22':
            plt.errorbar(xlow, data_by_size['mean_train'],
                         yerr=data_by_size['std_train'], fmt='D', label=r'Training')
            plt.errorbar(xhigh, data_by_size['mean_deploy'],
                         yerr=data_by_size['std_deploy'], fmt='X')
            plt.errorbar(xpoints + 0.1, random_results['mean'],
                         yerr=random_results['stddev'], fmt='P', color='darkgreen',
                         )
            plt.ylim(-4500, -20)
            plt.text(x=0.7, y=-45, s='A2C, 22 nodes', bbox=props)
            ax.tick_params(labelbottom=False, direction='in', which='both')
        elif isize == '55':
            plt.errorbar(xlow, data_by_size['mean_train'],
                         yerr=data_by_size['std_train'], fmt='D')
            plt.errorbar(xhigh, data_by_size['mean_deploy'],
                         yerr=data_by_size['std_deploy'], fmt='X', label=r'Real Network')
            plt.errorbar(xpoints + 0.1, random_results['mean'],
                         yerr=random_results['stddev'], fmt='P', color='darkgreen',
                         )
            plt.ylabel('Mean Rewards', fontweight='bold')
            plt.ylim(-6000, -280)
            plt.text(x=0.7, y=-510, s='PPO, 55 nodes', bbox=props)
            ax.tick_params(labelbottom=False, direction='in', which='both')
        elif isize == '60':
            plt.errorbar(xlow, data_by_size['mean_train'],
                         yerr=data_by_size['std_train'], fmt='D')
            plt.errorbar(xhigh, data_by_size['mean_deploy'],
                         yerr=data_by_size['std_deploy'], fmt='X')
            plt.errorbar(xpoints + 0.1, random_results['mean'],
                         yerr=random_results['stddev'], fmt='P', color='darkgreen',
                         label=r'Random Agent')
            plt.xticks(xpoints, order, rotation=30)
            plt.ylim(-6000, -30)
            plt.text(x=0.7, y=-80, s='DQN, 60 nodes', bbox=props)

        # use a y-log scale
        plt.yscale('symlog')
        plt.legend(loc='upper right')

    # adjust the image to not have gaps
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # save the figure
    plt.savefig('real_deployment.png', dpi=500)
    plt.savefig('real_deployment.pdf', dpi=500)
