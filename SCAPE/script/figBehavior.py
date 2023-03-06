import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from utils import buildBehaviorFigures as buildfig
from utils import boutClustering as bc

### Initialise: set path for all datasets

master_path = '/network/lustre/iss02/wyart/analyses/mathilde.lapoix/SCAPE/analysis_001/'
clusteringName = 'SCAPE_data_220227_5clusters'
# df_bout_path = '/home/mathilde.lapoix/Téléchargements/df_bout_all(1).pkl'
df_bout_path = master_path + '/df_bout_all.pkl'
# df_bout_path = master_path + '/df_bout_all_with_classification{}.pkl'.format(clusteringName)
df_bout_all = buildfig.load_df_bout_all(df_bout_path)
#  df_summary = pd.read_csv('/home/mathilde.lapoix/Téléchargements/df_summary.csv')
df_summary = pd.read_csv(master_path + '/df_summary.csv')
figPath = '/network/lustre/iss02/wyart/analyses/mathilde.lapoix/SCAPE/analysis_001/'
nFish = len(df_bout_all.Fishlabel.unique())
nFishEnucleated = len(df_bout_all[df_bout_all.enucleated == 1].Fishlabel.unique())
nFishEyes = len(df_bout_all[df_bout_all.enucleated == 0].Fishlabel.unique())
nRun = len(df_bout_all.Trial.unique())
nBouts = len(df_bout_all)

## Reattribute to each bout the bout cat assigned by the clustering

df_clustering, df_pca = bc.load_clustering_results('/home/mathilde.lapoix/anaconda3/envs/ZebraZoom/lib/python3.8/'
                                                   'site-packages/zebrazoom/dataAnalysis/resultsClustering/'
                                                   '{}/savedRawData/boutParameters.pkl'.format(clusteringName))
df_bout_all['classification'] = pd.Series(df_bout_all.index).apply(bc.reassign_bout_cluster,
                                                                   args=(df_bout_all, df_clustering))
df_bout_all.classification = df_bout_all.classification.fillna('not_assigned')
df_bout_all.to_pickle(master_path + '/df_bout_all_with_classification{}.pkl'.format(clusteringName))

## Panel A: Experimental set-up (manually drawn)

## Panel B: Example tail angle from both conditions (from already plotted figure with all ta)

#  Panel C: zoom in time spent swimming for each conditions

nfish = len(df_summary[~np.isnan(df_summary.prop_time_swimming)].fishlabel.unique())
nexp = len(df_summary[~np.isnan(df_summary.prop_time_swimming)])
plt.figure()
plt.title('{} fish, {} exp'.format(nfish, nexp))
sns.kdeplot(data=df_summary, x='prop_time_swimming', hue='enucleated', common_norm=False)
plt.savefig(figPath + '/time_swim_eye.svg')
plt.figure()
plt.title('{} fish, {} exp'.format(nfish, nexp))
sns.kdeplot(data=df_summary, x='prop_time_swimming', hue='stage', common_norm=False)
plt.savefig(figPath + '/time_swim_stage.svg')

# only - 6 dpf

a = df_summary[(df_summary.stage == '6 dpf') & ~(np.isnan(df_summary.prop_time_swimming))]
nfish = len(a[~np.isnan(a.prop_time_swimming)].fishlabel.unique())
nexp = len(a[~np.isnan(a.prop_time_swimming)])
plt.figure()
plt.title('{} fish, {} exp, only 6 dpf fish'.format(nfish, nexp))
sns.kdeplot(data=a, x='prop_time_swimming', hue='enucleated', common_norm=False)
plt.savefig(figPath + '/time_swim_eye_6dpf.svg')

params = ['Bout_Duration', 'Number_Osc', 'abs_Max_Bend_Amp', 'mean_TBF',
          'median_iTBF', 'max_iTBF', 'median_bend_amp', 'mean_tail_angle']

sns.set(style="darkgrid")


def violin_params_noOutliers(df_bout_all, params, title='All exp'):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12), num=title)

    nFish = len(df_bout_all.Fishlabel.unique())
    nFishEye = len(df_bout_all[df_bout_all.enucleated == 0].Fishlabel.unique())
    nFishEnu = len(df_bout_all[df_bout_all.enucleated == 1].Fishlabel.unique())
    nBouts = len(df_bout_all)

    fig.suptitle('{} fish (eyes: {}, no eyes: {}), {} bouts\nno outliers'.format(nFish,
                                                                                 nFishEye,
                                                                                 nFishEnu,
                                                                                 nBouts))
    for i, param in enumerate(params):
        ax_i = ax.flatten()[i]
        temp_df = df_bout_all[(df_bout_all[param] < df_bout_all[param].quantile(0.99)) &
                              (df_bout_all[param] > df_bout_all[param].quantile(0.01)) &
                              (df_bout_all.flag != 1)]  # remove outliers & flagged bouts
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
    fig.savefig(figPath + '/violinplot_noOutliers_' + title + '.svg')


def violin_params(df_bout_all, params):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('6dpf only, n={} fish'.format(len(df_bout_all[df_bout_all.stage == '6 dpf'].Fishlabel.unique())))
    for i, param in enumerate(params):
        ax_i = ax.flatten()[i]
        sns.violinplot(data=df_bout_all[df_bout_all.stage == '6 dpf'],
                       y=param, x='enucleated', ax=ax_i, palette="Pastel1")
        for x_pos in [0, 1]:
            n_obs = len(df_bout_all[df_bout_all.enucleated == x_pos][param].notna())
            median_pos = df_bout_all[df_bout_all.enucleated == x_pos][param].median()
            ax_i.text(x_pos, median_pos + 0.5, 'n:' + str(n_obs),
                      horizontalalignment='center',
                      size='small',
                      color='w',
                      weight='semibold')
    fig.savefig(figPath + '/violin_params_condition_6dpf.svg')

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

    fig.savefig(figPath + '/violin_params_stage.svg')


violin_params_noOutliers(df_bout_all, params)
violin_params_noOutliers(df_bout_all[df_bout_all.stage == '6 dpf'], params, '6_dpf_only')


#  Speed regimes


def plot_speed_regimes(df_bout_all, title):
    nFish = len(df_bout_all.Fishlabel.unique())
    nFishEye = len(df_bout_all[df_bout_all.enucleated == 0].Fishlabel.unique())
    nFishEnu = len(df_bout_all[df_bout_all.enucleated == 1].Fishlabel.unique())
    nBouts = len(df_bout_all)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, num=title)
    fig.suptitle('{} fish (eyes: {}, no eyes{}), {} bouts'.format(nFish,
                                                                  nFishEye,
                                                                  nFishEnu,
                                                                  nBouts))
    sns.kdeplot(data=df_bout_all[df_bout_all.flag != 1],
                x='median_iTBF', hue='enucleated', common_norm=False,
                ax=ax[0])
    sns.kdeplot(data=df_bout_all[df_bout_all.flag != 1],
                x='mean_TBF', hue='enucleated', common_norm=False,
                ax=ax[1])
    ax[0].set_xlim(5, 30)
    fig.savefig(figPath + '/' + title + '.svg')


def plot_speed_regimes_split_by_angle(df_bout_all, title):
    nFish = len(df_bout_all.Fishlabel.unique())
    nFishEye = len(df_bout_all[df_bout_all.enucleated == 0].Fishlabel.unique())
    nFishEnu = len(df_bout_all[df_bout_all.enucleated == 1].Fishlabel.unique())
    nBouts = len(df_bout_all)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, num=title)
    fig.suptitle('{} fish (eyes: {}, no eyes{}), {} bouts\nMax bend < 30 (up)\nMax Bend > 30 (down)'.format(nFish,
                                                                                                            nFishEye,
                                                                                                            nFishEnu,
                                                                                                            nBouts))
    sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp < 30) & (df_bout_all.flag != 1)],
                x='median_iTBF', hue='enucleated', common_norm=False,
                ax=ax[0, 0])
    sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp < 30) & (df_bout_all.flag != 1)],
                x='mean_TBF', hue='enucleated', common_norm=False,
                ax=ax[0, 1])
    ax[0, 0].set_xlim(5, 30)
    sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp > 30) & (df_bout_all.flag != 1)],
                x='median_iTBF', hue='enucleated', common_norm=False,
                ax=ax[1, 0])
    sns.kdeplot(data=df_bout_all[(df_bout_all.abs_Max_Bend_Amp > 30) & (df_bout_all.flag != 1)],
                x='mean_TBF', hue='enucleated', common_norm=False,
                ax=ax[1, 1])
    fig.savefig(figPath + '/' + title + '.svg')


plot_speed_regimes(df_bout_all, 'speed_regimes')
plot_speed_regimes(df_bout_all[df_bout_all.stage == '6 dpf'], 'speed_regimes_6dpf')


#  AMp regimes


def plot_amp_regimes(df_bout_all):
    plt.figure()
    plt.title('All bouts, n={}\n(w eyes:{}, w/o: {}\n{} runs'.format(nBouts,
                                                                     nFishEyes,
                                                                     nFishEnucleated,
                                                                     nRun))
    sns.histplot(data=df_bout_all[df_bout_all.Number_Osc >= 1.5],
                 x='abs_Max_Bend_Amp', hue='enucleated', common_norm=False, stat='probability')
    plt.savefig(figPath + '/max_bend_amp_condition.svg')


plot_amp_regimes(df_bout_all)

# dispersion of the behavior in the 2 conditions

df_summary['std_median_bend_amp'] = np.nan
df_summary['std_median_iTBF'] = np.nan
df_summary['std_ax_bend_amp'] = np.nan
for i in df_bout_all.Fishlabel.unique():
    df_summary.loc[df_summary.fishlabel == i, 'std_median_bend_amp'] = np.std(
        df_bout_all[df_bout_all.Fishlabel == i].median_bend_amp)
    df_summary.loc[df_summary.fishlabel == i, 'std_median_iTBF'] = np.std(
        df_bout_all[df_bout_all.Fishlabel == i].median_iTBF)
    df_summary.loc[df_summary.fishlabel == i, 'std_max_bend_amp'] = np.std(
        df_bout_all[df_bout_all.Fishlabel == i].abs_Max_Bend_Amp)

fig, ax = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle('All bouts, n={}\n(w eyes:{}, w/o: {})\n{} runs'.format(nBouts,
                                                                     nFishEyes,
                                                                     nFishEnucleated,
                                                                     nRun))
sns.kdeplot(data=df_summary, x='std_median_iTBF', hue='enucleated', common_norm=False, ax=ax[0])
sns.kdeplot(data=df_summary, x='std_median_bend_amp', hue='enucleated', common_norm=False, ax=ax[1])
sns.kdeplot(data=df_summary, x='std_max_bend_amp', hue='enucleated', common_norm=False, ax=ax[2])
fig.savefig(figPath + '/dispersion_std.svg')

#  Panel e: clustering
# n = 3 clusters, without outliers give us mostly 2 types of forward, 1 type of turns

df = bc.build_df_bout_trace_cluster(df_bout_all, nTimeSteps=80)

plt.figure(num='all_bouts')
sns.lineplot(data=df, x='time_point', y='tail_angle', hue='classification')
plt.title('All stages, all conditions mixed')
plt.savefig(figPath + 'cluster_lineplot_all_stages_all_cond.svg')
fig, ax = plt.subplots(1, 2, figsize=(12, 4), num='per_condition')
fig.suptitle('All stages, per condition')
sns.lineplot(data=df[df.condition == 1], x='time_point', y='tail_angle', hue='classification', ax=ax[0])
ax[0].set_title('Enucleated larvae')
sns.lineplot(data=df[df.condition == 0], x='time_point', y='tail_angle', hue='classification', ax=ax[1])
ax[1].set_title('Intact larvae')
fig.savefig(figPath + 'cluster_lineplot_all_stages_per_cond.svg')
fig.suptitle('6 dpf only, per condition')
sns.lineplot(data=df[df.condition == 1], x='time_point', y='tail_angle', hue='classification', ax=ax[0])
sns.lineplot(data=df[df.condition == 0], x='time_point', y='tail_angle', hue='classification', ax=ax[1])
fig.savefig(figPath + 'cluster_lineplot_6dpf_per_cond.svg')

# e1: typical tail angle trace for each + std

# TODO: add it directly into clusteringAnalysis pipeline

# e2: proportions of each cluster in each condition

# TODO: get nicer figure in the ZZ output, and add svg format

#  Panel f: comparison of kinematics for each condition

# TODO: get nicer figure in the ZZ output, and add svg format

# bout wise vs fish wise ?

# build example dff movie during behavior

exp_id = 79
ta = np.load(df_summary.savePath[exp_id] + df_summary.run[exp_id] + '/tail_angle.npy')
time_indices = np.arange(len(ta)) / 300
frames = (132906, 142515)
fig, ax = plt.subplots()
fig.suptitle('Exp {}, example tail angle trace between frames {} and {}'.format(exp_id, frames[0], frames[1]))
ax.plot(time_indices[frames[0]:frames[1]], ta[frames[0]:frames[1]])
ax.set_ylabel('Tail angle [°]')
ax.set_xlabel('Time [s]')
plt.savefig(figPath + '/ex_tail_angle_exp{}.svg'.format(exp_id))

# In example fish, check proportions of bouts in each cluster, and plot bouts in each cluster

df_exp = df_bout_all[df_bout_all.exp == 64].copy()
df_trace_exp = bc.build_df_bout_trace_cluster(df_exp, nTimeSteps=80)

nbouts = len(df_exp)
for cluster in df_exp.classification.unique():
    nbouts_cluster = len(df_exp[df_exp.classification == cluster])
    prop = int(100 * (nbouts_cluster / nbouts))
    print('{} bouts in cluster {} ({}/{})'.format(prop, cluster, nbouts_cluster, nbouts))

fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharey=True)
classifications = list(df_trace_exp.classification.unique())
for i, cluster in enumerate([0, 1, 2, 3, 4, 'not_assigned']):
    sns.lineplot(data=df_trace_exp[df_trace_exp.classification == cluster], x='time_point', y='tail_angle',
                 hue='bout_index', ax=ax.flatten()[i])
    ax.flatten()[i].set_title(cluster)
plt.tight_layout()
plt.savefig(
    '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/PhD_manuscript/Results/material/_F4_run2_clustering.svg')

#  With manual cat

dict_bout_types = {'forward': df_bout_all.abs_Max_Bend_Amp <= 25,
                   'left_turns': (df_bout_all.abs_Max_Bend_Amp > 25) & (df_bout_all.abs_Max_Bend_Amp < 60) & (
                           df_bout_all.Max_Bend_Amp > 0),
                   'right_turns': (df_bout_all.abs_Max_Bend_Amp > 25) & (df_bout_all.abs_Max_Bend_Amp < 60) & (
                           df_bout_all.Max_Bend_Amp < 0),
                   'others': (df_bout_all.abs_Max_Bend_Amp >= 60)}

df_bout_all['manual_cat'] = np.nan
for i, (key, mask) in enumerate(dict_bout_types.items()):
    df_bout_all.loc[mask, 'manual_cat'] = key

sns.set(style="darkgrid")
fig, ax = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Enucleated, 6 dpf only, no outliers')
fig_intact, ax_intact = plt.subplots(3, 3, figsize=(12, 12))
fig_intact.suptitle('intact, 6 dpf only, no outliers')
for i, param in enumerate(params):
    temp_df = df_bout_all[(df_bout_all[param] < df_bout_all[param].quantile(0.99)) &
                          (df_bout_all[param] > df_bout_all[param].quantile(0.01)) &
                          (df_bout_all.flag != 1) &
                          (df_bout_all.stage == '6 dpf')]
    ax_i = ax.flatten()[i]
    sns.violinplot(data=temp_df[temp_df.enucleated == 1],
                   y=param, x='manual_cat', ax=ax_i, palette="Pastel1")
    ax_i_intact = ax_intact.flatten()[i]
    sns.violinplot(data=temp_df[temp_df.enucleated == 0],
                   y=param, x='manual_cat', ax=ax_i_intact, palette="Pastel1")

fig.savefig(figPath + '/violin_params_manual_cat.svg')

# Line plot tail angle

df = bc.build_df_bout_trace_cluster(df_bout_all, nTimeSteps=80)
for param in ['flag', 'Fishlabel', 'manual_cat']:
    for i in df.bout_index.unique():
        flag = df_bout_all[param].iloc[i]
        df.loc[df.bout_index == i, param] = flag

palette = ['#FF6600', '#FF00FF', '#00FF00', '#808080']
fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
fig.suptitle('6 dpf only, per condition')
for condition, i in {'intact': 0, 'enucleated': 1}.items():
    temp_df = df[(df.condition == i) & (df.stage == '6 dpf') & (df.flag != 1)]
    sns.lineplot(data=temp_df,
                 ci='sd', hue_order=['forward', 'left_turns', 'right_turns', 'others'],
                 palette=sns.color_palette(palette),
                 x='time_point', y='tail_angle', hue='manual_cat',
                 ax=ax[i])
    ax[0].set_title('{} larvae, n={} larvae, n={} bouts'.format(condition,
                                                                len(temp_df.Fishlabel.unique()),
                                                                len(temp_df.bout_index.unique())))
temp_df = df[(df.condition == 1) & (df.stage == '6 dpf') & (df.flag != 1)]

fig.savefig(figPath + 'manual_cat_lineplot_6dpf_per_cond.svg')

# Summmary fish behavior enucleated and 6dpf
filt_df = df[(df.condition == 1) & (df.stage == '6 dpf')]
chosen_fish = filt_df.fishlabel.unique()
paths = df_bout_all[(df_bout_all.Fishlabel.isin(chosen_fish))].path.unique()
tail_angles = [pd.read_pickle(i)]
for fish in chosen_fish:
    fig, ax = plt.subplots(1, 1, figsize=(15, 6), sharex=True, sharey=True)
    fig.suptitle('6 dpf & enucleated:'+fish)
    temp_df = filt_df[filt_df.fishlabel == fish]
    sns.lineplot(data=temp_df,
                 ci='sd', hue_order=['forward', 'left_turns', 'right_turns', 'others'],
                 palette=sns.color_palette(palette),
                 x='time_point', y='tail_angle', hue='manual_cat',
                 ax=ax)
    ax.set_title('n={} bouts'.format(len(temp_df.bout_index.unique())))
    fig.savefig(figPath + '_' + fish + '_manual_cat_lineplot.svg')

    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('6 dpf & enucleated:' + fish)
    for i, key in enumerate(['forward', 'left_turns', 'right_turns']):
        n_bouts = len(temp_df[temp_df.manual_cat == key].bout_index.unique())
        sns.lineplot(data=temp_df[temp_df.manual_cat == key],
                     x='time_point', y='tail_angle', hue='manual_cat', units='bout_index',
                     estimator=None, ax=ax[i])
        ax[i].set_title('{}, n={}/{}'.format(key, n_bouts, len(temp_df.bout_index.unique())))
        ax[i].set_ylim(-65, 65)
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(figPath + '_' + fish + '_manual_cat_lineplot_no_estimator.svg')
# TO DO: ADD SUBPLOT WITH INDIVIDUAL TAIL ANGLE TRACES FOR EACH CAT

# Assymetry of tail movement in 6 dpf, enucleated fish


def manual_cat(i, df):
    if 25 <=df.abs_Max_Bend_Amp.loc[i] < 60:
        if df.Max_Bend_Amp.loc[i] < 0:
            output = 'right_turns'
        else:
            output = 'left_turns'
    elif df.abs_Max_Bend_Amp.loc[i] < 25:
        output = 'forward'
    else:
        output = 'struggle'
    return output


df_bout_all['manual_cat'] = pd.Series(df_bout_all.index).apply(manual_cat, args=(df_bout_all,))

df = df_bout_all[(df_bout_all.enucleated == 1) & (df_bout_all.stage == '6 dpf')]

palette = {'forward': "#F46036", 'left_turns': "#372248", 'right_turns': "#006E90", 'struggle': "#BEBEBE"}
fig1, axes1 = plt.subplots(2, 6, figsize=(15, 7), sharex=True, sharey=True)
fig1.suptitle('6dpf, enucleated larvae\nfirst bend amp')
fig2, axes2 = plt.subplots(2, 6, figsize=(15,7), sharex=True, sharey=True)
fig2.suptitle('6dpf, enucleated larvae\nsecond bend amp')
ax1 = axes1.flatten()
ax2 = axes2.flatten()
for i, fish in enumerate(df.Fishlabel.unique()):
    ax1[i].set_title(fish+'_'+str(len(df[df.Fishlabel == fish]))+'bouts')
    ax2[i].set_title(fish+'_'+str(len(df[df.Fishlabel == fish]))+'bouts')
    sns.kdeplot(data=df[df.Fishlabel == fish], x='First_Bend_Amp', ax=ax1[i],
                hue='manual_cat',common_norm=False, palette=palette)
    sns.kdeplot(data=df[df.Fishlabel == fish], x='Second_Bend_Amp', ax=ax2[i],
                hue='manual_cat',common_norm=False, palette=palette)
    for j in ax1:
        j.set_xlim(-100,100)
    for j in ax2:
        j.set_xlim(-100, 100)
    if i != len(df.Fishlabel.unique())-1:
        ax1[i].get_legend().remove()
        ax2[i].get_legend().remove()
fig1.tight_layout()
fig1.savefig('/network/lustre/iss02/wyart/analyses/mathilde.lapoix/SCAPE/analysis_001/first_bend_amp_kde.svg')
fig2.tight_layout()
fig2.savefig('/network/lustre/iss02/wyart/analyses/mathilde.lapoix/SCAPE/analysis_001/second_bend_amp_kde.svg')

# For all fish, show traces of bout category

df_traces = pd.read_pickle('/network/lustre/iss02/wyart/analyses/mathilde.lapoix/SCAPE/analysis_001/df_traces_6dpf_enucleated.pkl')
df = df_traces
palette = ['#FF6600', '#FF00FF', '#00FF00', '#808080']
filt_df = df[(df.condition == 1) & (df.stage == '6 dpf')]
chosen_fish = filt_df.fishlabel.unique()

plt.close('all')
for fish in chosen_fish:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)
    fig.suptitle('6 dpf & enucleated:' + fish)
    temp_df = filt_df[filt_df.fishlabel == fish]
    sns.lineplot(data=temp_df,
                 ci='sd', hue_order=['forward', 'left_turns', 'right_turns', 'others'],
                 palette=sns.color_palette(palette),
                 x='time_point', y='tail_angle', hue='manual_cat',
                 ax=ax)
    ax.set_ylim(-65, 65)
    ax.set_title('n={} bouts'.format(len(temp_df.bout_index.unique())))
    fig.savefig(figPath + 'Cat_single_fish/' + fish + '_manual_cat_lineplot.svg')

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('6 dpf & enucleated:' + fish)
    for i, key in enumerate(['forward', 'left_turns', 'right_turns']):
        n_bouts = len(temp_df[temp_df.manual_cat == key].bout_index.unique())
        sns.lineplot(data=temp_df[temp_df.manual_cat == key],
                     x='time_point', y='tail_angle',
                     hue='manual_cat',
                     palette=sns.color_palette(palette), hue_order=['forward', 'left_turns', 'right_turns', 'others'],
                     units='bout_index',
                     estimator=None, ax=ax[i])
        ax[i].set_title('{}, n={}/{}'.format(key, n_bouts, len(temp_df.bout_index.unique())))
        ax[i].set_ylim(-65, 65)
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(figPath + 'Cat_single_fish/' + fish + '_manual_cat_lineplot_no_estimator.svg')

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)
sns.lineplot(data=filt_df[filt_df.manual_cat == 'forward'],
             ci='sd',
             hue='fishlabel',
             x='time_point', y='tail_angle',
             ax=ax)
ax.set_ylim(-25, 25)
fig.suptitle('Mean forward bout for all enucleated, 6dpf fish')
plt.savefig(figPath + '/Forward_mean_enucleated_6dpf.svg')

# bout rate and ibi

