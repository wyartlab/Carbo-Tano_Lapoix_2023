import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
import seaborn as sns

from utils.utils_behavior_dataset import run_ZZ_extraction
from utils.import_data import load_summary_csv


master_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/analysis_001/'
# summary_csv = pd.read_csv('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/summaryData.csv')
# summary_csv['df_include'] = 1


summary_csv = load_summary_csv(
    'https://docs.google.com/spreadsheets/d/1VHFmX8j8rfwDiKghT5tb0LxZfmB5_qrgqYmcJxmB7RE/edit#gid=1097839266')
summary_csv['prop_time_swimming'] = np.nan
params_to_add = ['enucleated', 'frameRateSCAPE', 'stage', 'laserPower', 'duration',
                 'includeBehavior', 'includeAnalysis']
dict_all = {}
# TO DO: add step to save df frame also
for i in summary_csv.index:
    print('\n')
    df_bout, df_frame = run_ZZ_extraction(i, summary_csv)
    if isinstance(df_bout, pd.core.frame.DataFrame):

        for key in params_to_add:
            df_bout[key] = summary_csv.loc[i, key]
        df_bout['exp'] = i

        dict_all[i] = df_bout
        summary_csv.loc[i, 'prop_time_swimming'] = df_bout.Bout_Duration.sum() / summary_csv.duration[i]

df_bout_all = pd.concat(dict_all, ignore_index=True)
del i
del dict_all

#  Build df for clustering with ZZ

short_summary = summary_csv.loc[df_bout_all.exp.unique()].copy()
df_for_clustering = pd.DataFrame({'path': ['/'.join(i.split('/')[:-2]) for i in short_summary.ZZ_path],
                                  'trial_id': [i.split('/')[-2] for i in short_summary.ZZ_path],
                                  'fq': [300] * len(short_summary),
                                  'pixelsize': 70,
                                  'condition': list(short_summary.enucleated),
                                  'genotype': ['WT'] * len(short_summary),
                                  'include': ['1'] * len(short_summary)},
                                 index=df_bout_all.exp.unique())


def get_enucleated_cond(i):
    if i == 0:
        return 'not_enucleated'
    else:
        return 'enucleated'


df_for_clustering['condition'] = ['['+get_enucleated_cond(i)+']' for i in df_for_clustering['condition']]
df_for_clustering.to_csv('/home/mathilde.lapoix/anaconda3/envs/ZebraZoom/lib/python3.8/site-packages/'
                         'zebrazoom/dataAnalysis/experimentOrganizationExcel/SCAPE_data_220227.csv')
summary_csv.to_csv(master_path + '/df_summary.csv')

# RUN CLUSTERING

# LOAD RESULTS FROM CLUSTERING

df = pd.read_pickle('/home/mathilde.lapoix/anaconda3/envs/ZebraZoom/lib/python3.8/site-packages/zebrazoom/dataAnalysis/'
                    'resultsClustering/SCAPE_data_220227_3clusters/savedRawData/'
                    'boutParameters.pkl')['dfParam']

# Explore distrubtion of kinematic parameters

params = ['Bout_Duration', 'Number_Osc', 'abs_Max_Bend_Amp', 'mean_TBF',
          'median_iTBF', 'max_iTBF', 'median_bend_amp', 'mean_tail_angle']

sns.set(style="darkgrid")
fig, ax = plt.subplots(3, 3, figsize=(12, 12))

fig.suptitle('n={} fish, w/o outliers'.format(len(df_bout_all.Fishlabel.unique())))
for i, param in enumerate(params):
    ax_i = ax.flatten()[i]
    temp_df = df_bout_all[(df_bout_all[param] < df_bout_all[param].quantile(0.99)) &
                          (df_bout_all[param] > df_bout_all[param].quantile(0.01)) &
                          (df_bout_all.flag != 1)] # remove outliers & flagged bouts
    sns.violinplot(data=temp_df, y=param, x='enucleated', ax=ax_i, palette="Pastel1")
    for x_pos in [0, 1]:
        n_obs = len(temp_df[temp_df.enucleated == x_pos][param].notna())
        median_pos = temp_df[temp_df.enucleated == x_pos][param].median()
        ax_i.text(x_pos, median_pos + 0.5, 'n:' + str(n_obs),
                  horizontalalignment='center',
                  size='small',
                  color='w',
                  weight='semibold')
    if param in ['mean_TBF', 'median_iTBF', 'max_iTBF']:
        ax_i.set_ylim(0, 40)

#  Speed regimes

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Max bend < 30')
sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp < 30) & (df_bout_all.flag != 1)],
            x='median_iTBF', hue='enucleated', common_norm=False,
            ax=ax[0])
sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp < 30) & (df_bout_all.flag != 1)],
            x='mean_TBF', hue='enucleated', common_norm=False,
            ax=ax[1])
ax[0].set_xlim(5, 40)
ax[1].set_xlim(5, 40)
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
fig1.suptitle('Max bend > 30')
sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp > 30) & (df_bout_all.flag != 1)],
            x='median_iTBF', hue='enucleated', common_norm=False,
            ax=ax1[0])
sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp > 30) & (df_bout_all.flag != 1)],
            x='mean_TBF', hue='enucleated', common_norm=False,
            ax=ax1[1])
ax1[0].set_xlim(5, 40)
ax1[1].set_xlim(5, 40)

#  AMp regimes

plt.figure()
plt.title('All bouts, n={}\n(w eyes:{}, w/o: {}'.format(len(df_bout_all),
                                                        len(df_bout_all[df_bout_all.enucleated == 0]),
                                                        len(df_bout_all[df_bout_all.enucleated == 1])))
sns.histplot(data=df_bout_all[df_bout_all.Number_Osc >= 1.5],
             x='abs_Max_Bend_Amp', hue='enucleated', common_norm=False, stat='probability')

fig, ax = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('5dpf only, n={} fish'.format(len(df_bout_all[df_bout_all.stage == '5 dpf'].Fishlabel.unique())))
for i, param in enumerate(params):
    ax_i = ax.flatten()[i]
    sns.violinplot(data=df_bout_all[df_bout_all.stage == '5 dpf'],
                   y=param, x='enucleated', ax=ax_i, palette="Pastel1")
    for x_pos in [0, 1]:
        n_obs = len(df_bout_all[df_bout_all.enucleated == x_pos][param].notna())
        median_pos = df_bout_all[df_bout_all.enucleated == x_pos][param].median()
        ax_i.text(x_pos, median_pos + 0.5, 'n:' + str(n_obs),
                  horizontalalignment='center',
                  size='small',
                  color='w',
                  weight='semibold')

sns.set(style="darkgrid")
fig, ax = plt.subplots(3, 3, figsize=(12, 12))
for i, param in enumerate(params):
    ax_i = ax.flatten()[i]
    sns.violinplot(data=df_bout_all, y=param, x='stage', ax=ax_i, palette="Pastel1")
    for x_pos, stage in enumerate(df_bout_all.stage.unique()):
        n_obs = len(df_bout_all[df_bout_all.stage == stage][param].notna())
        median_pos = df_bout_all[df_bout_all.stage == stage][param].median()
        ax_i.text(x_pos, median_pos + 0.5, 'n:' + str(n_obs),
                  horizontalalignment='center',
                  size='small',
                  color='w',
                  weight='semibold')

df_bout_all['abs_first_bend'] = np.abs(df_bout_all['First_Bend_Amp'])
df_bout_all['abs_second_bend'] = np.abs(df_bout_all['Second_Bend_Amp'])
plt.figure()
sns.scatterplot(data=df_bout_all, hue='enucleated', x='abs_first_bend', y='abs_second_bend')

# Save all tail angles


def get_color_enucleation(index, df_summary, color_enu='orange', color_eye='royalBlue'):
    if df_summary['enucleated'][index] == 1:
        color = color_enu
    else:
        color = color_eye
    return color


fig, ax = plt.subplots(figsize=(20,12))
nFish = len(df_bout_all.Fishlabel.unique())
nFishEnucleated = len(df_bout_all[df_bout_all.enucleated == 1].Fishlabel.unique())
nRun = len(df_bout_all.Trial.unique())
fig.suptitle('All tail angles for {} larvae, {} runs\n({} enucleated)'.format(nFish, nRun, nFishEnucleated))

for i, j in enumerate(df_bout_all.exp.unique()):
    trial = df_bout_all.loc[df_bout_all.exp == j, 'Trial'].unique()[0]
    path = summary_csv.loc[summary_csv.index == j, 'savePath'].item()
    label = summary_csv['fishlabel'][j] + '_' + summary_csv['run'][j]
    ta = np.load(path + '/' + trial +'/tail_angle.npy')
    time_indices = np.arange(len(ta))/300
    ax.plot(time_indices, ta-i*50, color=get_color_enucleation(j, summary_csv), label=label)
plt.legend()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Tail angle [°]')
fig.savefig(master_path + '/all_tail_angles.svg')

# TO DO: save all tail angles
df_bout_all.to_csv(master_path + 'df_bout_all.csv')
df_bout_all.to_pickle(master_path + 'df_bout_all.pkl')
