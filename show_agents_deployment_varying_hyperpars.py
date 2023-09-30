"""
    This code shows how to process and plot the relevant information gathered
    after evaluating the agents performances in known scenarios.

"""

import numpy as np
import pandas as pd
import glob
import sys
import os
import matplotlib.pyplot as plt

from RL_plottings import plot_evaluation_performance
from RL_utils import process_df, get_summary_statistics, process_random_data

# Staging
current_dir = os.getcwd()

output_dir = os.path.join(current_dir, 'networks')
logs_dir = os.path.join(current_dir, 'logs_dir')
plot_dir = os.path.join(current_dir, 'figures')
data_dir = os.path.join(current_dir, 'results_data')

# Load the figure style, this is optional
plt.style.use(os.path.join(plot_dir, 'figure_style.mplstyle'))

# First generate the locations for each algorithm

xlocs = [i*3+1 for i in range(9)] # Do it for the 9 algorithms (3x3)
# instantiate the liss
xlow, xmid, xhig = [], [], []
for i in xlocs:
    xlow.append(i - 0.5)
    xmid.append(i - 0.)
    xhig.append(i + 0.5)

# Now I have the locations of the various algorithms hyper pars modificato

# now load the data
data_file = os.path.join(data_dir, 'agents_performances_deployed_noscenario.csv')

# load random agent performances
random_data = os.path.join(data_dir, 'random_agents_performances_training.csv')

# read the data
data =  pd.read_csv(data_file, header=0) # skip the header

# clean the data
cleaned_df = process_df(data)
new_df = cleaned_df.drop(columns='model').copy()

# process random agent data
random_results = pd.read_csv(random_data, header=0)

# get the random scores
y_random, y_stddev = process_random_data(random_results, 'std')

# now process the summary statistics
sum_stat = get_summary_statistics(new_df)

# this is ready to be passed to the plotter function
plot_evaluation_performance(sum_stat, xlocs,
                            xlow, xmid, xhig,
                            y_random, y_stddev,
                            'Training performances',
                            )
plt.grid()  # plot the grid
plt.tight_layout()
plt.show()

# To save the plots
# plt.savefig(os.path.join(figures, 'agents_training_performances_evaluation.png'), dpi=300)