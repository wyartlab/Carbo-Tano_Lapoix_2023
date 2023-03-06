import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pyabf
import seaborn as sns
import shelve
import sys
from scipy.stats import zscore


df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/data_summary_BH.csv')
fishlabel, plane = '210121_F05', '70um_bh'
fig_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/ML_pipeline_output/fig6/'+fishlabel+'_'+plane
print(fishlabel, plane)
output_path = df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == plane), 'output_path'].item()

shelve_out = shelve.open(output_path + '/shelve_calciumAnalysis.out')

old_cells = shelve_out['cells']
dff = shelve_out['dff_f_lp_inter']
dff_non_inter = shelve_out['dff']
bad_frames = shelve_out['bad_frames']
dff_non_inter[:,bad_frames] = np.nan
df_bouts = pd.read_pickle(output_path + '/dataset/df_bout')
df_frame = pd.read_pickle(output_path + '/dataset/df_frame')
tail_angle = shelve_out['tail_angle']
stim_trace = shelve_out['stim_trace']
time_trace_stim = shelve_out['time_trace_stim']
shift = shelve_out['shift']
noise = shelve_out['noise_f_lp']
stat = shelve_out['stat']
fps_ci = shelve_out['fps']
fps_bh = shelve_out['fps_beh']
direction = df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == plane), 'direction'].item()
time_indices_2p = np.arange(dff.shape[1]) / fps_ci
time_indices_bh = np.arange(len(tail_angle)) / fps_bh
print(list(shelve_out.keys()))

shelve_out.close()

cells = []
sc_bulbar = df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == plane), 'sc_bulbar'].item()
for cell in old_cells:
    if direction == 1:

        if stat[cell]['med'][1] > sc_bulbar:
            if (fishlabel == '210203_F03') & (plane == '70um_04'):
                if stat[cell]['med'][0] < 60:
                    print('Removed cell out of fish:', cell)
                else:
                    cells.append(cell)
            else:
                cells.append(cell)

    elif direction == 0:
        if stat[cell]['med'][0] < sc_bulbar:
            cells.append(cell)

print('Kept {} hindbrain cells out of {}'.format(len(cells), len(old_cells)))

pop_clustering_labels = np.load(fig_path + '/pop_clustering_labels.npy')

cmap_name = 'tab10'
cmap = matplotlib.cm.get_cmap(cmap_name)

colors = [cmap(i / 10) for i in range(len(set(pop_clustering_labels)))]
for label in set(pop_clustering_labels):
    plt.figure(figsize=(15, 10))
    plt.plot(time_indices_bh, tail_angle + 100, 'k')
    plt.title('In cluster ' + str(label), y=1.05)
    plt.xlabel('Time [s]')
    plt.ylabel('DFF')
    for i, cell in enumerate(np.array(cells)[pop_clustering_labels == label]):
        plt.plot(time_indices_2p, zscore(dff[cell, :], nan_policy='omit')*10 - i * 20, label='cell ' + str(cell),
                 color=colors[label])
    plt.grid(b=None)

    sum_trace = np.mean(dff[np.array(cells)[pop_clustering_labels == label]], axis=0)
    plt.plot(time_indices_2p, zscore(sum_trace, nan_policy='omit')*10 - (i + 1) * 20, label='summed trace',
             color='silver')
    plt.plot(time_trace_stim-shift, stim_trace-(i+2)*20,
             label='stim', color='coral')
    plt.savefig(fig_path + '/zscore_calcium_trace_cluster' + str(label) + '.svg')

    # non interpolated dff

    plt.figure(figsize=(15, 10))
    plt.plot(time_indices_bh, tail_angle + 100, 'k')
    plt.title('In cluster ' + str(label), y=1.05)
    plt.xlabel('Time [s]')
    plt.ylabel('DFF')
    for i, cell in enumerate(np.array(cells)[pop_clustering_labels == label]):
        plt.plot(time_indices_2p, zscore(dff_non_inter[cell, :], nan_policy='omit')*10 - i * 20, label='cell ' + str(cell),
                 color=colors[label])
    plt.grid(b=None)

    sum_trace = np.mean(dff_non_inter[np.array(cells)[pop_clustering_labels == label]], axis=0)
    plt.plot(time_indices_2p, zscore(sum_trace, nan_policy='omit')*10 - (i + 1) * 20, label='summed trace',
             color='silver')
    plt.plot(time_trace_stim-shift, stim_trace-(i+2)*20,
             label='stim', color='coral')
    plt.savefig(fig_path + '/zscore_non_interp_calcium_trace_cluster' + str(label) + '.svg')
# fig, ax = plt.subplots()
