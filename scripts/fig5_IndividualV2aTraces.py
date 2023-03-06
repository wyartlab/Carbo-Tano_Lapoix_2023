"""
date: 20/12/2021
author: mathildelpx
input: dataset with calcium activity of V2a in paralysed larvae, during MLR single shot stimulation.
output: traces of each individually identified V2a in pontine and retropontine regions.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from scipy.stats import zscore

plt.style.use('seaborn-poster')

df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/data_summary_II.csv')
df_final = pd.read_pickle('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/Datasets/Paralysed/df_final.pkl')
save_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/Figures/material/V2a_individual_neurons_traces/'

exp_ids = [3, 18, 19, 22, 23]
cells_per_exp = {'3': [6, 7, 8, 15],
                 '18': [5, 17],
                 '19': [21, 35, 45, 50],
                 '22': [0, 2, 5, 6],
                 '23': [8, 9, 15, 17, 21, 28, 30, 54, 71]}

for i in exp_ids:
    fish, plane = df_summary.fishlabel[i], df_summary.plane[i]
    output_path = df_summary.output_path[i]
    with open(output_path + '/dataset/struct', 'rb') as handle:
        struct = pickle.load(handle)
        dff_f = struct['dff_filtered']
        cells = struct['cells']
        stims_start = struct['stims_start']
        stim_trace = struct['stim_trace']
        time_indices = struct['time_indices']
        fps = struct['frame_rate']

    print('\n{}, {}'.format(fish, plane))

    start, stop = stims_start[0] - 10, stims_start[5]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle('{}_{}'.format(fish, plane))

    for j in cells_per_exp[str(i)]:
        try:
            group = df_final.loc[(df_final.fishlabel == fish) &
                                 (df_final.plane == plane) &
                                 (df_final.cell == j), 'final_cell_group'].unique().item()
        except ValueError:
            group = 'unknown_group(not_kept_during_analysis)'
        ax.plot(time_indices[start:stop],
                zscore(dff_f[j, start:stop], nan_policy='omit') + 1,
                label='cell_{}_{}'.format(j, group))

    ax.plot(time_indices[start:stop], stim_trace[start:stop],
            color='silver',
            label='stim')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('zscore DF/F')
    ax.legend(bbox_to_anchor=(1.05, 1))

    fig.savefig(save_path + '{}_{}.svg'.format(fish, plane))
