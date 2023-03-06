"""

author: mathildelpx
creationTime: 22/02/2022
goal: load calcium imaging data output from suite2p, behavior pre-processed data and compute map of recruitment and
regressor analysis.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import zscore
from random import sample
import os
import pickle
from glob import glob
import json

os.chdir('/home/mathilde.lapoix/PycharmProjects/SCAPE/')

import utils.recruitmentCellBout as rct
import utils.import_data as load
import utils.createExpClass as createExpClass
import utils.createCellClass as createCellClass
import utils.processSuite2pOutput as processs2p
import utils.functionalClustering as functionalClustering
import utils.modelNeuronalActivity as ml
from utils.tools_processing import build_mean_image

# %% Initialise

summary_csv = load.load_summary_csv(
    'https://docs.google.com/spreadsheets/d/1VHFmX8j8rfwDiKghT5tb0LxZfmB5_qrgqYmcJxmB7RE/edit#gid=1097839266')
exp_id = 58
reload_Exp = False
savefig = False

# %% Load pre-processed info

if reload_Exp:
    with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Exp.pkl', 'rb') as f:
        Exp = pickle.load(f)

else:
    Exp = createExpClass.Run(summary_csv, exp_id)
print(Exp.savePath, Exp.runID)

df_frame, df_bout = load.load_behavior_dataframe(Exp)
ta = np.array(df_frame.Tail_angle)
time_indices_bh = np.array(df_frame.Time_index)
setattr(Exp, 'tail_angle', ta)

# %% Process calcium traces

Exp.load_suite2p_outputs()

for plane in Exp.suite2pData.keys():

    Exp.correct_suite2p_outputs(plane)
    Exp.filter_f(plane)
    # f_corrected = Exp.suite2pData[plane]['F_corrected_filter']

    Exp.suite2pData[plane]['spks_corrected'] = processs2p.runSpksExtraction(Exp.suite2pData[plane]['F_corrected'],
                                                                            Exp=Exp,
                                                                            batch_size=1000,
                                                                            tau=1.8,
                                                                            fs=Exp.frameRateSCAPE.copy())

    Exp.suite2pData[plane]['dff'], Exp.suite2pData[plane]['noise'] = processs2p.calc_dff(
        Exp.suite2pData[plane]['cells'],
        Exp.suite2pData[plane]['F_corrected'],
        bad_frames=Exp.bad_frames)

    zscore_dff = np.zeros(Exp.suite2pData[plane]['F_corrected'].shape)
    for cell in Exp.suite2pData[plane]['cells']:
        zscore_dff[cell,] = zscore(Exp.suite2pData[plane]['dff'][cell,], nan_policy='omit')
    Exp.suite2pData[plane]['zscore_dff'] = zscore_dff

Exp.assign_behavior_trace(df_frame)
Exp.assign_time_indices()

try:
    os.mkdir(os.path.join(Exp.savePath + Exp.runID, 'plots'))
except FileExistsError:
    pass
setattr(Exp, 'fig_path', os.path.join(Exp.savePath + Exp.runID, 'plots'))

# %% Assign midline per plane

try:
    with open(Exp.savePath + Exp.runID + '/dict_midline.pkl', 'rb') as f:
        dict_midline = pickle.load(f)
    setattr(Exp, 'midline_lim', dict_midline)
except FileNotFoundError:
    Exp.define_midline_per_plane()
    with open(Exp.savePath + Exp.runID + '/dict_midline.pkl', 'wb') as f:
        pickle.dump(Exp.midline_lim, f)

#  %% Load normalised positions and build overall background

setattr(Exp, 'limits_crop', pd.read_csv(Exp.savePath + Exp.runID + '/limits_crop.csv'))
Exp.build_mean_image()

# %% Save Exp object

# with open(Exp.savePath + Exp.runID + '/Exp.pkl', 'wb') as f:
#     pickle.dump(Exp, f)

# %% Create Cell Objects

if reload_Exp:
    with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Cells.pkl', 'rb') as f:
        Cells = pickle.load(f)

### Compute cell info

cell_id = 0
Cells = []

for plane in Exp.suite2pData.keys():
    cells = Exp.suite2pData[plane]['cells']

    for cell in cells:
        Cells.append(createCellClass.Cell(exp=Exp, plane=plane, cell=cell, id=cell_id))
        cell_id += 1

del cell_id

print('Total number of cells: {}'.format(len(Cells)))

# for cell positions

dict_x_group = {'spinal_cord': (0, 160),
                'caudal_medulla': (160, 260),
                'rostral_medulla': (260,340),
                'retropontine': (340,402),
                'pontine': (402, 464),
                'prepontine': (464, 507)}

_ = [Cell.assign_cell_group(dict_x=dict_x_group, dorsoventral_limit=86) for Cell in Cells]
_ = [Cell.assign_side(Exp) for Cell in Cells]
_ = [Cell.compute_norm_y_pos(Exp) for Cell in Cells]

# Correct cell signals from motion artifacts
_ = [Cell.mask_and_fill_signal(Exp) for Cell in Cells]

with open(Exp.savePath + Exp.runID + '/Cells.pkl', 'wb') as f:
    pickle.dump(Cells, f)

# %% Translate behavior trace

ml.build_behavior_traces(Exp)
ml.build_resampled_behavior_traces(Exp)
ml.build_pure_forward(Exp, df_bout=df_bout)

# %% Test recruitment

# for cell in sample(range(len(Cells)), 5):
#     fig, ax = plt.subplots()
#     fig.suptitle('Cell {}'.format(cell))
#     ax.plot(Exp.time_indices_SCAPE, zscore(Cells[cell].dff, nan_policy='omit'), color='darkgreen')
#     ax.plot(Exp.time_indices_SCAPE, Cells[cell].spks, color='grey')
#     ax.plot(time_indices_bh, ta / 30, color='black')

# %% Compute recruitment

## build dataframe with cell and bout info

if reload_Exp:
    df_recruitment = pd.read_pickle(Exp.savePath + Exp.runID + '/df_recruitment.pkl')
else:
    df_recruitment = rct.build_df_recruitment(Cells, df_bout, Exp)

    df_recruitment.to_pickle(Exp.savePath + Exp.runID + '/df_recruitment.pkl')

### Compute median and mean recruitment for all bout types

keys = ['mean_recruitment_f', 'median_recruitment_f', 'mean_max_dff_f', 'median_max_dff_f']

MASKS = {'f': df_recruitment.abs_Max_Bend_Amp < 25,
         'l': (df_recruitment.abs_Max_Bend_Amp >= 25) & (df_recruitment.abs_Max_Bend_Amp < 60) & (
                 df_recruitment.mean_tail_angle > 0),
         'r': (df_recruitment.abs_Max_Bend_Amp >= 25) & (df_recruitment.abs_Max_Bend_Amp < 60) & (
                 df_recruitment.mean_tail_angle < 0)}

for bout_type in MASKS.keys():
    df_recruitment = rct.getStatsActivityDuringBout(df_recruitment, bout_type, Exp, MASKS)

df_noDupl = df_recruitment.copy()
df_noDupl = df_noDupl.drop_duplicates(subset='cell_id')

COLOR_VAR = ['mean_recruitment', 'median_recruitment', 'mean_max_dff', 'median_max_dff']
for plane in Exp.suite2pData.keys():
    nTypes = len(MASKS.keys())
    nVars = len(COLOR_VAR)
    fig, ax = plt.subplots(nTypes, nVars, num=plane, figsize=(16, 16))
    for i, colorVar in enumerate(COLOR_VAR):
        rct.plot_maps_recruitment_bout_type(plane, colorVar, MASKS, Exp, df_noDupl, ax[:, i])

### Recruitment during locomotion ?

dict_bout_types = {'forward': df_bout.abs_Max_Bend_Amp <= 25,
                   'left_turns': (df_bout.abs_Max_Bend_Amp > 25) & (df_bout.abs_Max_Bend_Amp < 60) & (
                           df_bout.Max_Bend_Amp > 0),
                   'right_turns': (df_bout.abs_Max_Bend_Amp > 25) & (df_bout.abs_Max_Bend_Amp < 60) & (
                           df_bout.Max_Bend_Amp < 0),
                   'others': (df_bout.abs_Max_Bend_Amp >= 60)}


def build_df_trace_bout_type(df_bout, Exp, dict_bout_types, results_path=None,
                             nTimeSteps=80):
    df_bout['bout_type'] = np.nan
    for bout_type, mask in dict_bout_types.items():
        df_bout.loc[mask, 'bout_type'] = bout_type

    nBouts = len(df_bout)
    time_steps = np.arange(nTimeSteps) / 300
    output = pd.DataFrame({'bout_index': np.repeat(df_bout.NumBout, nTimeSteps),
                           'tail_angle': np.nan,
                           'time_point': np.tile(time_steps, nBouts),
                           'frame': np.tile(range(nTimeSteps), nBouts),
                           'bout_type': np.repeat(df_bout.bout_type, nTimeSteps)})

    # Prior to looping into all bouts, load all struct so we don't have to reload them
    if results_path is None:
        results_path = glob(Exp.savePath + Exp.runID + '/results_*.txt')[0]
    with open(results_path) as f:
        struct = json.load(f)['wellPoissMouv'][0][0]

    for bout_num in df_bout.NumBout:

        #  load tail angle for this bout
        bout_trace = np.array(struct[bout_num]['TailAngle_smoothed']) * 57.2958

        #  check that length is more than 70 frames, or correct it
        if len(bout_trace) < nTimeSteps:
            bout_trace_temp = np.zeros((nTimeSteps))
            bout_trace_temp[:] = np.nan
            bout_trace_temp[0:len(bout_trace)] = bout_trace
            bout_trace = bout_trace_temp

        output.loc[output.bout_index == bout_num, 'tail_angle'] = bout_trace[0:nTimeSteps]

    return output


df_traces = build_df_trace_bout_type(df_bout, Exp, dict_bout_types, nTimeSteps=80)
df_traces.to_pickle(os.path.join(Exp.savePath, Exp.runID, 'df_traces.pkl'))

plt.figure(num='all_bouts')
sns.lineplot(data=df_traces, x='time_point', y='tail_angle', hue='bout_type')
plt.title('Manual category applied on bouts')
plt.xlabel('Time [s]')
plt.savefig(Exp.savePath + Exp.runID + '/lineplot_bout_types.svg')
fig, ax = plt.subplots(3, 1, figsize=(12, 10))
for i, key in enumerate(dict_bout_types.keys()):
    n_bouts = len(df_traces[df_traces.bout_type == key].bout_index.unique())
    sns.lineplot(data=df_traces[df_traces.bout_type == key],
                 x='time_point', y='tail_angle', hue='bout_type', units='bout_index',
                 estimator=None, ax=ax[i])
    ax[i].set_title('{}, n={}/{}'.format(key, n_bouts, len(df_traces.bout_index.unique())))
    ax[i].set_ylim(-65, 65)
plt.xlabel('Time [s]')
plt.tight_layout()
plt.savefig(Exp.savePath + Exp.runID + '/lineplot_bout_types_no_estimator.svg')

df_recruitment = rct.more_active_during_locomotion(Cells, Exp, df_bout, df_recruitment, dict_bout_types, p0=0.05)
df_recruitment = rct.more_active_during_bout_types_reciprocal(Cells, Exp, df_bout,
                                                              df_recruitment, dict_bout_types, p0=0.05)

# Plot example distribution for forward greater cells
rct.plot_ex_forward_cells(Cells, Exp, df_bout, df_recruitment, savefig=True)

# Map all recruitment
# rct.map_recruitment_single_bout_type(Exp, df_recruitment, 'left_turns', savefig=True, coef_hue=True)
rct.map_all_recruitment_comparisons(Exp, df_recruitment, savefig=savefig)

#  Map cross activation
rct.map_cross_activation_during_bouts_types(Exp,
                                            df_recruitment,
                                            palettes=None,
                                            savefig=savefig)

palettes = {'L-F': ["#987284"],
            'R-F': ["#75B9BE"]}

f_spe, l_spe, r_spe = rct.get_list_cells_active_during_bout_types(df_recruitment)

combinations = {'L-F': set.difference(*map(set, [l_spe, f_spe])),
                'R-F': set.difference(*map(set, [r_spe, f_spe])), }

df_short = df_recruitment.drop_duplicates('cell_id').copy()
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('Cells diff-active during bout types')

for i, combination in enumerate(combinations.keys()):
    cells = combinations[combination]
    ax[i, 0].set_title('{}\nDorsal view'.format(combination))
    ax[i, 0].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
    sns.scatterplot(data=df_short[~df_short.cell_id.isin(cells)], x='x_pos', y='norm_y_pos',
                    hue='bout_id',
                    alpha=0.25,
                    palette='Greys',
                    ax=ax[i, 0])
    sns.scatterplot(data=df_short[df_short.cell_id.isin(cells)], x='x_pos', y='norm_y_pos',
                    hue='bout_id',
                    alpha=0.5,
                    palette=sns.color_palette(palettes[combination]),
                    ax=ax[i, 0])

    ax[i, 1].set_title('Sagittal view')
    ax[i, 1].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
    sns.scatterplot(data=df_short[~df_short.cell_id.isin(cells)], x='x_pos', y='plane',
                    hue='bout_id',
                    alpha=0.25,
                    palette='Greys',
                    ax=ax[i, 1])
    sns.scatterplot(data=df_short[df_short.cell_id.isin(cells)], x='x_pos', y='plane',
                    hue='bout_id',
                    alpha=0.5,
                    palette=sns.color_palette(palettes[combination]),
                    ax=ax[i, 1])
plt.savefig(Exp.fig_path + '/difference_active_turns_forward.svg')

# Save

with open(Exp.savePath + Exp.runID + '/Cells.pkl', 'wb') as f:
    pickle.dump(Cells, f)

with open(Exp.savePath + Exp.runID + '/Exp.pkl', 'wb') as f:
    pickle.dump(Exp, f)

df_recruitment.to_pickle(Exp.savePath + Exp.runID + '/df_recruitment.pkl')

#  Clusters of neurons

# remove single isolated neurons from the plot for 220127_4_run2
# forward_component_neurons = df_recruitment[(df_recruitment.more_active_during_bout_type_forward) &
#                                            (df_recruitment.more_active_during_bout_type_left_turns) &
#                                            (df_recruitment.more_active_during_bout_type_right_turns) &
#                                            (df_recruitment.x_pos < 310) &
#                                            (df_recruitment.x_pos > 40)].cell_id.unique()
forward_component_neurons = df_recruitment[(df_recruitment.more_active_during_bout_type_forward) &
                                           (df_recruitment.more_active_during_bout_type_left_turns) &
                                           (df_recruitment.more_active_during_bout_type_right_turns)].cell_id.unique()

mask_group = df_recruitment.group == 'pontine_ventral'
mask_turn = (
            df_recruitment.more_active_during_bout_type_left_turns | df_recruitment.more_active_during_bout_type_right_turns)
mask_not_f = ~df_recruitment.more_active_during_bout_type_forward
# mask_ventral = df_recruitment.plane > 180
mask_ventral = df_recruitment.plane > 65
steering_cluster_vP = df_recruitment[mask_group & mask_turn & mask_not_f & mask_ventral].cell_id.unique()
print(len(steering_cluster_vP))
mask_group = df_recruitment.group == 'retropontine_ventral'
mask_turn = (
            df_recruitment.more_active_during_bout_type_left_turns | df_recruitment.more_active_during_bout_type_right_turns)
mask_not_f = ~df_recruitment.more_active_during_bout_type_forward
steering_cluster_RP = df_recruitment[mask_group & mask_turn & mask_not_f].cell_id.unique()

no_cluster = set([i.cellID for i in Cells]).difference(set().union(forward_component_neurons,
                                                                   steering_cluster_RP,
                                                                   steering_cluster_vP))

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].imshow(Exp.mean_background, cmap='Greys_r')
ax[1].imshow(Exp.mean_background, cmap='Greys_r')

sns.scatterplot(data=df_recruitment[df_recruitment.cell_id.isin(no_cluster)].drop_duplicates('cell_id'),
                x='x_pos',
                y='norm_y_pos',
                cmap='Greys',
                hue='bout_id',
                ax=ax[0])
sns.scatterplot(data=df_recruitment[df_recruitment.cell_id.isin(no_cluster)].drop_duplicates('cell_id'),
                x='x_pos',
                y='plane',
                ax=ax[1])

for cluster in [forward_component_neurons, steering_cluster_RP, steering_cluster_vP]:
    sns.scatterplot(data=df_recruitment[df_recruitment.cell_id.isin(cluster)].drop_duplicates('cell_id'),
                    x='x_pos',
                    y='norm_y_pos',
                    ax=ax[0])
    sns.scatterplot(data=df_recruitment[df_recruitment.cell_id.isin(cluster)].drop_duplicates('cell_id'),
                    x='x_pos',
                    y='plane',
                    ax=ax[1])
plt.savefig(Exp.fig_path + '/identified_clusters.svg')

# clustering forward spe cells

forward_component = df_recruitment[
    (df_recruitment.more_active_during_bout_type_left_turns) & (df_recruitment.more_active_during_bout_type_forward) & (
        df_recruitment.more_active_during_bout_type_right_turns)].cell_id.unique()
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

forward_components_traces = np.vstack([Cells[i].F_corrected for i in forward_component])
corr_fc = pd.DataFrame(forward_components_traces).T.corr()


def plot_raster(neuron_traces, vmax, ylabel):
    n_neurons = neuron_traces.shape[0]
    plt.figure(figsize=(15, int(n_neurons / 10) + 1))
    plt.imshow(neuron_traces, aspect="auto", vmin=0, vmax=vmax)
    plt.ylabel(ylabel)
    # ax[0].set_yticks(np.arange(neuron_names.shape[0]))
    # ax[0].set_yticklabels(neuron_names)
    plt.colorbar()
    plt.show()


zscore_traces = np.vstack([zscore(Cells[i].F_corrected, nan_policy='omit') for i in forward_component])
n_clusters = 4
clustering = AgglomerativeClustering(n_clusters=n_clusters, ).fit(corr_fc)
dend = dendrogram(linkage(corr_fc, method='ward'))
labels = clustering.labels_

for i in range(n_clusters):
    plot_raster(zscore_traces[labels == i], n_clusters, 'cluster ' + str(i))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(Exp.mean_background, cmap='Greys_r')
ax[1].imshow(Exp.mean_background, cmap='Greys_r')
colors = ['blue', 'cyan', 'magenta', 'red', 'orange']
for i in range(n_clusters):
    cells = forward_component[labels == i]
    for cell in cells:
        ax[0].plot(Cells[cell].x_pos, Cells[cell].norm_y_pos, 'o', color=colors[i])
        ax[1].plot(Cells[cell].x_pos, Cells[cell].plane, 'o', color=colors[i])

plt.figure()
plt.plot(Exp.time_indices_bh, Exp.tail_angle)

## Poisson reg to model cell activity

dict_modeling = {'TA': Exp.ta_resampled,
                 'TA_left': Exp.ta_left_resampled,
                 'TA_right': Exp.ta_right_resampled,
                 'iTBF': np.array(ml.build_freq_array(Exp, df_frame)),
                 'increase_TA': ml.compute_absolute_change(Exp.ta_resampled, 3)}
modeling_results = {}


def run_GLM(x, feature, Cells,
            mask=np.where(np.logical_and(Exp.ta_resampled != 0, Exp.ta_resampled < 60)),
            model='GLM_Poisson', p0=0.01, savefig=False):
    results = []
    for Cell in Cells:
        results.append(ml.model_spks_given_behavior_GLM(Cell.spks[mask], behavior_trace=x[mask]))

    ml.map_sig_modelled(results, Cells, Exp, df_recruitment,
                        model_used=model, feature_used=feature, p0=p0, savefig=savefig)
    return results


for feature, x in dict_modeling.items():
    modeling_results[feature] = run_GLM(x, feature, Cells, savefig=False)

# proportion of cells in each group explained

clusters = {'forward_component': forward_component_neurons,
            'steering_RP': steering_cluster_RP,
            'steering_vP': steering_cluster_vP}

pvalue_cluster_explained_by_iTBF = {'forward_component': [], 'steering_RP': [], 'steering_vP': []}
for name, cluster in clusters.items():
    a = []
    for cell in cluster:
        a.append(modeling_results['iTBF'][cell].pvalues[0])
    pvalue_cluster_explained_by_iTBF[name] = a
    nsig = len(np.where(np.array(a) < 0.01)[0])
    ncells = len(cluster)
    prop = round(100 * nsig / ncells, 2)
    print('\n\n {}:\n explained by iTBF: {}/{} neurons ({}%)\nmean pvalue:{}'.format(name,
                                                                                     nsig,
                                                                                     ncells,
                                                                                     prop,
                                                                                     np.mean(a)))

pvalue_cluster_explained_by_iTBF = {'forward_component': [], 'steering_RP': [], 'steering_vP': []}
for name, cluster in clusters.items():
    for side in df_recruitment.side.unique():
        a = []
        for cell in cluster:
            if df_recruitment[df_recruitment.cell_id == cell].side.unique()[0] == side:
                a.append(modeling_results['TA_{}'.format(side)][cell].pvalues[0])
        nsig = len(np.where(np.array(a) < 0.01)[0])
        ncells = len(a)
        prop = round(100 * nsig / ncells, 2)
        print('\n\n {}:\n explained by iTA_{}: {}/{} neurons on ipsi side ({}%)\nmean pvalue:{}'.format(name,
                                                                                                        side,
                                                                                                        nsig,
                                                                                                        ncells,
                                                                                                        prop,
                                                                                                     np.mean(a)))
        a = []
        for cell in cluster:
            if df_recruitment[df_recruitment.cell_id == cell].side.unique()[0] != side:
                a.append(modeling_results['TA_{}'.format(side)][cell].pvalues[0])
        nsig = len(np.where(np.array(a) < 0.01)[0])
        ncells = len(a)
        prop = round(100 * nsig / ncells, 2)
        print('\n\n {}:\n explained by iTA_{}: {}/{} neurons on contra side ({}%)\nmean pvalue:{}'.format(name,
                                                                                                        side,
                                                                                                        nsig,
                                                                                                        ncells,
                                                                                                        prop,
                                                                                                        np.mean(a)))
## %% Correlation between pairs of neurons

from scipy import signal

## %% Compute functional groups

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib


# all_positions = np.load(Exp.savePath + Exp.runID + '/all_positions.npy', allow_pickle=True)
# all_background = np.load(Exp.savePath + Exp.runID + '/all_background.npy', allow_pickle=True).item()
#
# functionalClustering.build_neuron_traces(Exp, Cells)
#
# # cmap_name = 'tab10'
# # cmap = matplotlib.cm.get_cmap(cmap_name)
# #  colors = [cmap(i/10) for i in range(nClusters)]
#
# corr_dff = functionalClustering.get_corr_matrix(Exp.dff_traces, cross_corr=True, lags=2)
# labels = functionalClustering.runAggloClustering_vizu(nClusters=5,
#                                                       cmap=matplotlib.cm.get_cmap('tab10'),
#                                                       corr_dff=corr_dff,
#                                                       vmax=5,
#                                                       Exp=Exp,
#                                                       Cells=Cells,
#                                                       root_figPath=Exp.savePath + Exp.runID)

# plot functional clustering


# from scipy.stats import pearsonr
#
# df_corr_clusters = pd.DataFrame({'cluster': np.arange(nClusters),
#                                  'corr_vigor': np.nan,
#                                  'corr_left': np.nan,
#                                  'corr_right': np.nan,
#                                  'corr_forward': np.nan}, index=np.arange(nClusters))
# df_corr_clusters_spks = pd.DataFrame({'cluster': np.arange(nClusters),
#                                  'corr_vigor': np.nan,
#                                  'corr_left': np.nan,
#                                  'corr_right': np.nan,
#                                  'corr_forward': np.nan}, index=np.arange(nClusters))
#
# for label in np.arange(nClusters):
#     x = pd.DataFrame(np.mean(neuron_traces[labels == label], axis=0))
#     x = np.array(x.interpolate()[0])
#     x2 = pd.DataFrame(np.mean(spks_traces[labels == label], axis=0))
#     x2 = np.array(x2.interpolate()[0])
#
#     df_corr_clusters.loc[label, 'corr_vigor'], _ = pearsonr(tail_angle_resampled, x)
#     df_corr_clusters.loc[label, 'corr_left'], _ = pearsonr(ta_left_resampled, x)
#     df_corr_clusters.loc[label, 'corr_right'], _ = pearsonr(ta_right_resampled, x)
#     df_corr_clusters.loc[label, 'corr_forward'], _ = pearsonr(ta_forward_resampled, x)
#
#
#     df_corr_clusters_spks.loc[label, 'corr_vigor'], _ = pearsonr(tail_angle_resampled, x2)
#     df_corr_clusters_spks.loc[label, 'corr_left'], _ = pearsonr(ta_left_resampled, x2)
#     df_corr_clusters_spks.loc[label, 'corr_right'], _ = pearsonr(ta_right_resampled, x2)
#     df_corr_clusters_spks.loc[label, 'corr_forward'], _ = pearsonr(ta_forward_resampled, x2)
#
#
# ## For each cell, correlation to each model
#
# ### Compute correlation to motor parameters
#
# df_correlation = pd.DataFrame({'cell_id': [i.cellID for i in Cells],
#                                'cell_num': [i.init_cellID for i in Cells],
#                                'x_pos': [i.x_pos for i in Cells],
#                                'y_pos': [i.y_pos for i in Cells],
#                                'plane': [i.plane for i in Cells],
#                                'group': [i.group for i in Cells],
#                                'corr_power': np.nan,
#                                'corr_left': np.nan,
#                                'corr_right': np.nan,
#                                'corr_forward': np.nan,
#                                'p_power': np.nan,
#                                'p_left': np.nan,
#                                'p_right': np.nan,
#                                'p_forward': np.nan},
#                               index=np.arange(len(Cells)))
#
# for i in df_correlation.index:
#
#     df_correlation.loc[i, 'corr_power'], df_correlation.loc[i, 'p_power'] = pearsonr(tail_angle_resampled, Cells[i].spks)
#     df_correlation.loc[i, 'corr_left'], df_correlation.loc[i, 'p_left'] = pearsonr(ta_left_resampled, Cells[i].spks)
#     df_correlation.loc[i, 'corr_right'], df_correlation.loc[i, 'p_right'] = pearsonr(ta_right_resampled, Cells[i].spks)
#     df_correlation.loc[i, 'corr_forward'], df_correlation.loc[i, 'p_forward'] = pearsonr(ta_forward_resampled, Cells[i].spks)
#
#
# hue_max = df_correlation[['corr_power', 'corr_left', 'corr_right', 'corr_forward']].max().max()
# all_planes = list(all_background.keys())
# all_planes.sort()
# for mov in ['power', 'left', 'right', 'forward']:
#     col = 'corr_'+mov
#     fig, ax = plt.subplots(6, 5, figsize=(20, 15))
#     for i, key in enumerate(all_planes):
#         ax.flatten()[i].imshow(all_background[key], cmap='Greys')
#         sns.scatterplot(data=df_correlation[(df_correlation.plane == key) &
#                                             (df_correlation['p_'+mov] < 0.05)],
#                         x='y_pos', y='x_pos', hue=col, hue_norm=(-hue_max, hue_max),
#                         ax=ax.flatten()[i], palette='coolwarm')
#     plt.tight_layout()
#     fig.savefig(Exp.savePath + Exp.runID + '/sig_corr_to_{}.svg'.format(mov))


def calc_cross_corr(signal_1, signal_2, max_lags):
    nFrames = signal_1.shape[0]
    boundaries_lags = (int(nFrames / 2) - max_lags, int(nFrames / 2) + max_lags)
    correlation = signal.correlate(signal_1 - signal_1.mean(), signal_2 - signal_2.mean(), mode="full")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    corr_coef = correlation / (nFrames * np.nanstd(signal_1) * np.nanstd(signal_2))  # coef coefficient
    max_coef = max(corr_coef[boundaries_lags[0]:boundaries_lags[1]], key=abs)
    arg_max = np.nanargmax(np.abs(corr_coef[boundaries_lags[0]:boundaries_lags[1]]))
    max_lag = lags[boundaries_lags[0]:boundaries_lags[1]][arg_max]  # which lag gave the highest corr coef

    return max_coef, max_lag


from tqdm import tqdm


def get_cross_corr(Cells, n_lags):
    lags_matrix = np.zeros((len(Cells), len(Cells)))
    lags_matrix[:] = np.nan
    corr_matrix = np.zeros((len(Cells), len(Cells)))
    corr_matrix[:] = np.nan

    pairs_tested = []
    for i, Cell1 in tqdm(enumerate(Cells), desc='First neuron of the pair ', total=len(Cells)):
        for j, Cell2 in enumerate(Cells):

            if any([(j, i) == a for a in pairs_tested]):
                continue
            x = Cell1.spks
            y = Cell2.spks

            max_coef, max_lag = calc_cross_corr(x, y, n_lags)
            corr_matrix[i, j] = max_coef
            lags_matrix[i, j] = max_lag
            pairs_tested.append((i, j))

    return corr_matrix, lags_matrix



