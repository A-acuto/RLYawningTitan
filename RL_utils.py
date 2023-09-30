"""
    Collections of codes needed for preprocessing of the data
"""

import pandas as pd
import numpy as np

def process_df(datadf: pd.DataFrame) -> pd.DataFrame:
    """
        Basic cleaning of the dataframe not much generalisable for different purposes
    :param:
    datadf : dataframe containing the summary statistics of all agents

    :return:
    dataframe : reformatted to help the visualisation purposes
    """

    algo: list = []
    tp: list = []
    nodes: list = []

    for i in range(len(datadf)):
        datadf['reward'].iloc[i] = np.float16(datadf['reward'].iloc[i][1:-1])
        datadf['lenght'].iloc[i] = int(datadf['lenght'].iloc[i][1:-1])
        temp = datadf['model'].iloc[i]
        clean = temp.split('/')[-1].split('.')[0].split('\\')[-1]
        splet = clean.split('_')

        algo.append(splet[0])  # add the algorithm name
        if len(splet) >= 4:
            if 'DQN' in splet and 'std' in splet:
                tp.append(splet[1])  # type
                nodes.append(int(splet[2]))  # number of nodes
            elif 'std' in splet:
                tp.append(splet[1])
                nodes.append(int(splet[2]))
            else:
                tp.append(splet[1] + splet[2])
                nodes.append(int(splet[3]))

        if len(splet) == 3:
            tp.append(splet[1])
            nodes.append(int(splet[2]))

    return datadf.assign(**{'algorithm': algo, 'type': tp, 'nodes': nodes})


def get_summary_statistics(datadf: pd.DataFrame) -> dict:
    """
        create a summary statistics, from the Dataframe of the data processed get the unique size of
        networks, algorithms types and obtain the mean and the standard deviaions
        of the score performances
    :param:
        datadf : dataframe containing the scores of the agents for each trial
    :return:
        dictionary : of scores divided by network size
    """

    summary_statistics: dict = {}

    networks = list(datadf['nodes'].unique())  # get the unique network sizes

    for inet in networks:
        net1 = datadf[datadf['nodes'] == inet].copy()  # copy of the starting dataframe
        unique_type = net1['type'].unique()  # unique types and algorithmss
        unique_algo = net1['algorithm'].unique()

        sums0: dict = {}
        for ialgo in unique_algo:

            sums: dict = {}
            for itype in unique_type:
                rwds = net1[(net1['algorithm'] == ialgo) & (net1['type'] == itype)]['reward']
                lgts = net1[(net1['algorithm'] == ialgo) & (net1['type'] == itype)]['lenght']

                sums[itype] = {'rw_mean': np.mean(rwds),
                               'rw_med': np.median(rwds),
                               'rw_std': np.std(rwds),
                               'ln_mean': np.mean(lgts),
                               'ln_med': np.median(lgts),
                               'ln_std': np.std(lgts)}
            sums0[ialgo] = sums
        summary_statistics[str(inet)] = sums0

    return summary_statistics

def process_random_data(datadf: pd.DataFrame, extension: str) -> (list, list):
    """
        Auxuliary function to process the stored data for the random agent
        performances.
    :param:
    datadf : dataframe containing the random agent performances on all
             variations
    mod : string defining which extension look for
    :return:
    list of list: containing the mean scores and the standard deviation
                 of the performances
    """

    # trim the dataframe matching the specific extension
    trimmed = datadf.query(f'mod =="{extension}"')

    y_random = list(trimmed['mean'])
    std_random = list(trimmed['stddev'])

    return (y_random, std_random)