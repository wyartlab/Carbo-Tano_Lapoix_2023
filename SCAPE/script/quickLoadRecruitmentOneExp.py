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
exp_id = 65
savefig = False

# %% Load pre-processed info

with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Exp.pkl', 'rb') as f:
    Exp = pickle.load(f)

print(Exp.savePath, Exp.runID)

df_frame, df_bout = load.load_behavior_dataframe(Exp)

with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Cells.pkl', 'rb') as f:
        Cells = pickle.load(f)

df_recruitment = pd.read_pickle(Exp.savePath + Exp.runID + '/df_recruitment.pkl')

dict_bout_types = {'forward': df_bout.abs_Max_Bend_Amp <= 25,
                   'left_turns': (df_bout.abs_Max_Bend_Amp > 25) & (df_bout.abs_Max_Bend_Amp < 60) & (
                               df_bout.Max_Bend_Amp > 0),
                   'right_turns': (df_bout.abs_Max_Bend_Amp > 25) & (df_bout.abs_Max_Bend_Amp < 60) & (
                               df_bout.Max_Bend_Amp < 0),
                   'others': (df_bout.abs_Max_Bend_Amp >= 60)}

df = df_recruitment.drop_duplicates('cell_id')
all = 0
for group in df.group.unique():
    print('\n\n{}'.format(group))
    f_spe, l_spe, r_spe = rct.get_list_cells_active_during_bout_types(df[df.group == group])
    print('f spe: {}'.format(len(f_spe)))
    print('l spe: {}'.format(len(l_spe)))
    print('r spe: {}'.format(len(r_spe)))
    all_s = len(set.intersection(*map(set, [f_spe, l_spe, r_spe])))
    all += all_s
    print('f,l and r spe: {}'.format(all_s))


dict_modeling = {'TA': Exp.ta_resampled,
                 'TA_left': Exp.ta_left_resampled,
                 'TA_right': Exp.ta_right_resampled,
                 'iTBF': np.array(ml.build_freq_array(Exp, df_frame)),
                 'increase_TA': ml.compute_absolute_change(Exp.ta_resampled, 3)}
modeling_results = {}


p0 = 0.001

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
    modeling_results[feature] = run_GLM(x, feature, Cells, p0= p0, savefig=False)

for key in dict_modeling.keys():
    df_recruitment['sig_encoded_by_{}'.format(key)] = np.nan
    results = modeling_results[key]
    for i, Cell in enumerate(Cells):
        if (results[i].pvalues < p0) & (results[i].params[0] > 0):
            value = True
        else:
            value = False
        df_recruitment.loc[df_recruitment.cell_id == Cell.cellID, 'sig_encoded_by_{}'.format(key)] = value

for i in df_recruitment.group.unique():
    print('\n\n', i)
    print('group: ', i)
    n_cells_group = len(df_recruitment[df_recruitment.group == i].cell_id.unique())
    for j in modeling_results.keys():
        n_cells_recruited = len(df_recruitment[df_recruitment['sig_encoded_by_{}'.format(j)] == True].cell_id.unique())
        n_cells_recruited_group = len(df_recruitment[(df_recruitment['sig_encoded_by_{}'.format(j)] == True) & (
                    df_recruitment.group == i)].cell_id.unique())
        print('feature:', j)
        print('prop of cells encoded in this group: {}%'.format(100 * n_cells_recruited_group / n_cells_group))
        print('number of cells encoded in this group: {}/{} neurons'.format(n_cells_recruited_group, n_cells_group))
        print('prop of cells encoded that are in this group: {}%'.format(
            100 * n_cells_recruited_group / n_cells_recruited))

a = []
for plane in Exp.suite2pData.keys():
    stat = Exp.suite2pData[plane]['stat']
    cells = Exp.suite2pData[plane]['cells']
    count = 0
    for i in cells:
        if 'Manual' in stat[i].keys():
            count += 1

    a.append(100*count/len(cells))