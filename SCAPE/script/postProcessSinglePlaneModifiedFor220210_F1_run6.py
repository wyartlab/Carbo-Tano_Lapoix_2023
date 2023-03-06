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
import utils.modelNeuronalActivity as ml

# %% Initialise

summary_csv = load.load_summary_csv(
    'https://docs.google.com/spreadsheets/d/1VHFmX8j8rfwDiKghT5tb0LxZfmB5_qrgqYmcJxmB7RE/edit#gid=1097839266')
exp_id = 58

# %% Load pre-processed info

with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Exp.pkl', 'rb') as f:
    Exp = pickle.load(f)

print(Exp.savePath, Exp.runID)

df_frame, df_bout = load.load_behavior_dataframe(Exp)
ta = np.array(df_frame.Tail_angle)
time_indices_bh = np.array(df_frame.Time_index)

with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Cells.pkl', 'rb') as f:
    Cells = pickle.load(f)

# Remove bouts in the first part of the trace
start = 30000
ta = ta[start:]
df_frame = df_frame.iloc[start:,]
df_bout = df_bout[df_bout.BoutStart > start]

setattr(Exp, 'tail_angle', ta)

Exp.assign_behavior_trace(df_frame)
Exp.assign_time_indices()

# %% Translate behavior trace

ml.build_behavior_traces(Exp)
ml.build_resampled_behavior_traces(Exp)
ml.build_pure_forward(Exp, df_bout=df_bout)

# %% Compute recruitment

## build dataframe with cell and bout info

df_recruitment = rct.build_df_recruitment(Cells, df_bout, Exp)

### Compute median and mean recruitment for all bout types

keys = ['mean_recruitment_f', 'median_recruitment_f', 'mean_max_dff_f', 'median_max_dff_f']

MASKS = {'f': df_recruitment.abs_Max_Bend_Amp < 25,
         'l': (df_recruitment.abs_Max_Bend_Amp >= 25) & (df_recruitment.abs_Max_Bend_Amp < 60) & (
                 df_recruitment.mean_tail_angle > 0),
         'r': (df_recruitment.abs_Max_Bend_Amp >= 25) & (df_recruitment.abs_Max_Bend_Amp < 60) & (
                 df_recruitment.mean_tail_angle < 0)}

for bout_type in MASKS.keys():
    df_recruitment = rct.getStatsActivityDuringBout(df_recruitment, bout_type, Exp, MASKS)

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

        #  load tail angle for this bout
        bout_trace = np.array(struct[bout_num]['TailAngle_smoothed']) * 57.2958

        #  check that length is more than 70 frames, or correct it
        if len(bout_trace) < nTimeSteps:
            bout_trace_temp = np.zeros((nTimeSteps))
            bout_trace_temp[:] = np.nan
            bout_trace_temp[0:len(bout_trace)] = bout_trace
            bout_trace = bout_trace_temp

        output.loc[output.bout_index == bout_num, 'tail_angle'] = bout_trace[0:nTimeSteps]

    return output


df_traces = build_df_trace_bout_type(df_bout, Exp, dict_bout_types, nTimeSteps=80)
df_traces.to_pickle(os.path.join(Exp.savePath, Exp.runID, 'df_traces.pkl'))


def plot_df_trace(df_traces, Exp, dict_bout_types, savefig=True):
    plt.figure(num='all_bouts')
    sns.lineplot(data=df_traces, x='time_point', y='tail_angle', hue='bout_type')
    plt.title('Manual category applied on bouts')
    plt.xlabel('Time [s]')
    plt.savefig(Exp.savePath + Exp.runID + '/lineplot_bout_types.svg')
    fig, ax = plt.subplots(4, 1, figsize=(12, 10))
    for i, key in enumerate(dict_bout_types.keys()):
        n_bouts = len(df_traces[df_traces.bout_type == key].bout_index.unique())
        sns.lineplot(data=df_traces[df_traces.bout_type == key],
                     x='time_point', y='tail_angle', hue='bout_type', units='bout_index',
                     estimator=None, ax=ax[i])
        ax[i].set_title('{}, n={}/{}'.format(key, n_bouts, len(df_traces.bout_index.unique())))
        ax[i].set_ylim(-65, 65)
    plt.xlabel('Time [s]')
    plt.tight_layout()
    if savefig:
        plt.savefig(Exp.savePath + Exp.runID + '/lineplot_bout_types_no_estimator.svg')


plot_df_trace(df_traces, Exp, dict_bout_types)


df_recruitment = rct.more_active_during_locomotion(Cells, Exp, df_bout, df_recruitment, dict_bout_types, p0=0.01)
df_recruitment = rct.more_active_during_bout_types_reciprocal(Cells, Exp, df_bout,
                                                              df_recruitment, dict_bout_types, p0=0.01)

# Plot example distribution for forward greater cells
rct.plot_ex_forward_cells(Cells, Exp, df_bout, df_recruitment, savefig=True)

# Map all recruitment
# rct.map_recruitment_single_bout_type(Exp, df_recruitment, 'left_turns', savefig=True, coef_hue=True)
rct.map_all_recruitment_comparisons(Exp, df_recruitment, savefig=False)

#  Map cross activation
rct.map_cross_activation_during_bouts_types(Exp,
                                            df_recruitment,
                                            palettes=None,
                                            savefig=False)

# %% Map recruitment for each bout type


for bout_type in dict_bout_types.keys():
    try:
        rct.map_recruitment_single_bout_type(Exp, df_recruitment, 
                                             bout_type, 
                                             savefig=True, coef_hue=False)
    except ValueError:
        print('no map available for {}'.format(bout_type))
        continue

# %% Compute recruitment

# Save

df_recruitment.to_pickle(Exp.savePath + Exp.runID + '/df_recruitment_shorten.pkl')

