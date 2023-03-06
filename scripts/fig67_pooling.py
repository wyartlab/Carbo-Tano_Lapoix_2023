import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import shelve

plt.style.use('seaborn-poster')

summary_csv = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/data_summary_BH.csv')
data_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/ML_pipeline_output/fig6/'
save_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/ML_pipeline_output/fig6/Jan_2022/'
features = ['freq_abs', 'freq_change', 'amp_abs', 'amp_change']

dict_linReg = {}
neuron_id = 0
for exp in os.listdir(data_path):
    if os.path.isdir(data_path + exp):
        try:
            temp_df = pd.read_pickle(data_path + exp + '/df_linReg.pkl')
        except FileNotFoundError:
            continue
        fishlabel = exp.split('_')[0] + '_' + exp.split('_')[1]
        temp_df['fishlabel'] = fishlabel
        plane = exp.split('_')[2] + '_' + exp.split('_')[3]
        temp_df['plane'] = plane
        temp_df['exp'] = exp
        for neuron in temp_df.neuron.unique():
            temp_df.loc[temp_df.neuron == neuron, 'neuron_id'] = neuron_id
            neuron_id += 1

        if summary_csv[(summary_csv.fishlabel == fishlabel) &
                       (summary_csv.plane == plane)].useLinReg.unique().item() == 1:
            temp_df['useLinReg'] = True
        else:
            temp_df['useLinReg'] = False
        dict_linReg[exp] = temp_df

df_linReg_all = pd.concat(dict_linReg, ignore_index=True)

#  wide format

# df_wide = pd.DataFrame({'alpha0': list(df_linReg_all[df_linReg_all.feature_id == 0].alpha),
#                         'alpha1': list(df_linReg_all[df_linReg_all.feature_id == 1].alpha),
#                         'alpha2': list(df_linReg_all[df_linReg_all.feature_id == 2].alpha),
#                         'alpha3': list(df_linReg_all[df_linReg_all.feature_id == 3].alpha)})
#
# wcss = []  # store inertia of model
#
# for i in range(1, 30):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(df_wide)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 30), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
#
# n_clusters_kmean = 6
#
# kmeans = KMeans(n_clusters=n_clusters_kmean, init='k-means++', max_iter=300, n_init=10, random_state=0)
#
# results_kmean = kmeans.fit_predict(df_wide[['alpha0', 'alpha1', 'alpha2', 'alpha3']])
# df_wide['label'] = results_kmean
# df_linReg_all['kmean_label'] = np.tile(results_kmean, len(features))
#
# plt.style.use('seaborn-talk')
#
# fig, ax = plt.subplots(1, 2)
# sns.swarmplot(data=df_linReg_all, y='alpha', x='feature_label',
#               hue='kmean_label', ax=ax[0], palette='tab10')
# ax[0].legend(bbox_to_anchor=(1.2, 1))
#
# fig, ax = plt.subplots(1, n_clusters_kmean, sharex=True, sharey=True)
# for i in range(n_clusters_kmean):
#     sns.violinplot(data=df_linReg_all[df_linReg_all.kmean_label == i], y='alpha', x='feature_label', ax=ax[i])
#
# #  df_linReg_all.to_csv(data_path + '/df_linReg_all.csv')

# Load big df with norm position

df_all = pd.read_pickle('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/V2a_recruitment_behavior/'
                        'analysis_11/df_with_spinal_cord.pkl')
df_all = df_all[df_all.final_cell_group != 'spinal_cord']

df_linReg_all['norm_x_pos'] = np.nan
df_linReg_all['norm_y_pos'] = np.nan
df_linReg_all['approx_plane'] = np.nan
df_linReg_all['cell_id'] = np.nan

for neuron_id in df_linReg_all.neuron_id.unique():
    fish = df_linReg_all[df_linReg_all.neuron_id == neuron_id].fishlabel.unique().item()
    plane = df_linReg_all[df_linReg_all.neuron_id == neuron_id].plane.unique().item()
    approx_plane = plane.split('_')[0]
    cell = df_linReg_all[df_linReg_all.neuron_id == neuron_id].neuron.unique().item()
    norm_x = df_all.loc[(df_all.fishlabel == fish) & (df_all.plane == plane) &
                        (df_all.cell == cell)].norm_x_pos.unique().item()
    norm_y = df_all.loc[(df_all.fishlabel == fish) & (df_all.plane == plane) &
                        (df_all.cell == cell)].norm_y_pos.unique().item()
    cell_id = df_all.loc[(df_all.fishlabel == fish) & (df_all.plane == plane) &
                         (df_all.cell == cell)].cell_id.unique().item()
    df_linReg_all.loc[df_linReg_all.neuron_id == neuron_id, 'norm_x_pos'] = norm_x
    df_linReg_all.loc[df_linReg_all.neuron_id == neuron_id, 'norm_y_pos'] = norm_y
    df_linReg_all.loc[df_linReg_all.neuron_id == neuron_id, 'approx_plane'] = approx_plane
    df_linReg_all.loc[df_linReg_all.neuron_id == neuron_id, 'cell_id'] = cell_id

df_linReg_all.to_pickle(save_path + '/df_linReg_all.pkl')

ops_ref = np.load(
    '/network/lustre/iss01/wyart/rawdata/2pehaviour/MLR/Calcium_Imaging/210121/F05/70um_bh/suite2p/plane0/ops.npy',
    allow_pickle=True).item()

# check norm position

# for exp in os.listdir(data_path):
#     if os.path.isdir(data_path + exp):
#         fig, ax = plt.subplots()
#         ax.imshow(ops_ref['meanImg'], cmap='Greys')
#         sns.scatterplot(data=df_linReg_all[df_linReg_all.exp == exp],
#                         y='norm_x_pos', x='norm_y_pos', ax=ax)
#         fig.suptitle(exp)

# Plot positions of cells in the chosen clusters, for each experiment

fig, ax = plt.subplots(1, 1, figsize=(4, 6))
ax.imshow(ops_ref['meanImg'], cmap='Greys')
sns.scatterplot(data=df_linReg_all[df_linReg_all.feature_id == 0],
                x='norm_y_pos', y='norm_x_pos',
                hue='exp',
                linewidth=2, palette='Blues',
                ax=ax)
ax.legend(bbox_to_anchor=(1.1, 1))
fig.suptitle('All cells in chosen cluster')
fig.savefig(save_path + '/all_cells_chosen_cluster.svg')

# Plot forward activity index of cells in picked cluster only

df_all['in_chosen_cluster'] = False
df_all['useLinReg'] = False
for neuron_id in df_linReg_all.neuron_id.unique():
    fish = df_linReg_all[df_linReg_all.neuron_id == neuron_id].fishlabel.unique().item()
    plane = df_linReg_all[df_linReg_all.neuron_id == neuron_id].plane.unique().item()
    cell = df_linReg_all[df_linReg_all.neuron_id == neuron_id].neuron.unique().item()
    df_all.loc[(df_all.fishlabel == fish) &
               (df_all.plane == plane) &
               (df_all.cell == cell), 'in_chosen_cluster'] = True
    if df_linReg_all[df_linReg_all.neuron_id == neuron_id].useLinReg.unique().item():
        df_all.loc[(df_all.fishlabel == fish) &
                   (df_all.plane == plane) &
                   (df_all.cell == cell), 'useLinReg'] = True

## compute forward activity

df_all['pop_clustering'] = False
for exp in df_linReg_all.exp.unique():
    fish = exp.split('_')[0] + '_' + exp.split('_')[1]
    plane = exp.split('_')[2] + '_' + exp.split('_')[3]
    df_all.loc[(df_all.fishlabel == fish) & (df_all.plane == plane), 'pop_clustering'] = True

for i in df_all.cell_id.unique():
    r_S = df_all[df_all.cell_id == i].mean_recruitment_S_component.unique().item()
    r_F = df_all[df_all.cell_id == i].mean_recruitment_F_component.unique().item()
    if (r_F == 0) & (r_S == 0):
        output = np.nan
    else:
        output = (r_F - r_S) / (r_F + r_S)
    df_all.loc[df_all.cell_id == i, 'forward_activity_index'] = output

sns.displot(data=df_all[(df_all.syl == 0) & (df_all.pop_clustering)], x='forward_activity_index',
            hue='in_chosen_cluster', kind='kde', common_norm=False)
plt.savefig(save_path + 'kde_forward_activity_index.svg')

#  Plot duration encoding
from scipy.stats import linregress

for cell in set(df_all[df_all.in_chosen_cluster].cell_id):

    if len(df_all[(df_all.cell_id == cell) & (df_all.syl_cat == 'F')]) == 0:
        continue
    n_recruitment_f = len(
        df_all[(df_all.cell_id == cell) & (df_all.syl_cat == 'F') & (df_all.recruitment_f == 1)])
    if n_recruitment_f == len(df_all[(df_all.cell_id == cell) & (df_all.syl_cat == 'F')]):
        df_all.at[df_all.cell_id == cell, 'forward_specific_absolute'] = True

    # TODO: test difference between log and log10
    x = np.log(df_all[(df_all.cell_id == cell) & (df_all.syl_cat == 'F')].syl_n_osc)
    y = df_all[(df_all.cell_id == cell) & (df_all.syl_cat == 'F')].norm_max_dff_f

    slope, intercept, r, p, se = linregress(x, y)

    df_all.at[(df_all.cell_id == cell), 'slope_dff_log_osc_F'] = slope
    df_all.at[(df_all.cell_id == cell), 'pvalue_dff_log_osc_F'] = p

df_all['log_slope_df_log_osc_F'] = np.log10(df_all.slope_dff_log_osc_F)

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Log10 slope between norm max DF/F and log of number of oscillations')
ax.imshow(ops_ref['meanImg'], cmap='Greys')

sns.scatterplot(data=df_all[(df_all.slope_dff_log_osc_F.notna()) & (df_all.syl == 0)],
                x='norm_y_pos', y='norm_x_pos',
                hue='log_slope_df_log_osc_F',
                ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1))
fig.savefig(save_path + '/log_slope_dff_n_osc.svg')

# Map which cells had max DF/F were succesfully explained by number of oscillations
df_no_duplicates = df_all.drop_duplicates(subset='cell_id')
df_no_duplicates['group_oscillation_encoding'] = (df_no_duplicates['pvalue_dff_log_osc_F'] < 0.12) & \
                                                 (df_no_duplicates.slope_dff_log_osc_F >= 0.8)


def get_approx_plane(i):
    return i.split('_')[0]


df_no_duplicates['approx_plane'] = list(df_no_duplicates.plane.apply(get_approx_plane))
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(ops_ref['meanImg'], cmap='Greys')
sns.scatterplot(data=df_no_duplicates[df_no_duplicates.in_chosen_cluster],
                x='norm_y_pos', y='norm_x_pos', hue='group_oscillation_encoding',
                size='approx_plane',
                ax=ax)

ax.legend(bbox_to_anchor=(1.05, 1))
fig.savefig(save_path + '/groups_explained_or_not_by_nOsc.svg')
succesfull_duration_encoded_cell_id = list(df_no_duplicates[df_no_duplicates.in_chosen_cluster &
                                                            df_no_duplicates.group_oscillation_encoding].cell_id)

#  Map all alpha color and size encoded but only sig & positive
fig, ax = plt.subplots(1, 4, figsize=(12, 6))
for i, j in enumerate(df_linReg_all.feature_label.unique()):
    ax[i].imshow(ops_ref['meanImg'], cmap='Greys')
    sns.scatterplot(data=df_linReg_all[
        (df_linReg_all.feature_label == j) & (df_linReg_all.useLinReg) &
        (df_linReg_all.significant_pvalue) & (df_linReg_all.alpha > 0)],
                    x='norm_y_pos', y='norm_x_pos',
                    hue='alpha', hue_norm=(0, 40),
                    size='alpha', size_norm=(0, 40),
                    linewidth=2, palette='Blues',
                    ax=ax[i])
    ax[i].set_title(j)
    if i < 3:
        ax[i].get_legend().remove()
fig.savefig(save_path + '/sig_alpha_map_color_size_coded.svg')

#  Map all alpha, but split by plane
fig, ax = plt.subplots(2, 4, figsize=(12, 14))
fig.suptitle('70um (top) & 90 um (down)')
for i, j in enumerate(df_linReg_all.feature_label.unique()):
    ax[0, i].imshow(ops_ref['meanImg'], cmap='Greys')
    sns.scatterplot(data=df_linReg_all[
        (df_linReg_all.feature_label == j) & (df_linReg_all.useLinReg)
        & (df_linReg_all.significant_pvalue)
        & (df_linReg_all.alpha > 0) & (df_linReg_all.approx_plane == '70um')],
                    x='norm_y_pos', y='norm_x_pos',
                    hue='alpha', hue_norm=(0, 40),
                    size='alpha', size_norm=(0, 40),
                    linewidth=2, palette='Blues',
                    ax=ax[0, i])
    ax[0, i].set_title('70um_' + j)
    if i < 3:
        ax[0, i].get_legend().remove()

    ax[1, i].imshow(ops_ref['meanImg'], cmap='Greys')
    sns.scatterplot(data=df_linReg_all[
        (df_linReg_all.feature_label == j) & (df_linReg_all.useLinReg)
        & (df_linReg_all.significant_pvalue)
        & (df_linReg_all.alpha > 0) & (df_linReg_all.approx_plane == '90um')],
                    x='norm_y_pos', y='norm_x_pos',
                    hue='alpha', hue_norm=(0, 40),
                    size='alpha', size_norm=(0, 40),
                    linewidth=2, palette='Blues',
                    ax=ax[1, i])
    ax[1, i].set_title('90um_' + j)
    if i < 3:
        try:
            ax[1, i].get_legend().remove()
        except AttributeError:
            continue
fig.savefig(save_path + '/sig_alpha_map_plane_wise.svg')

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(ops_ref['meanImg'], cmap='Greys')
sns.scatterplot(data=df_linReg_all[(df_linReg_all.feature_id == 0) & (df_linReg_all.useLinReg)],
                x='norm_y_pos', y='norm_x_pos',
                ax=ax)
fig.savefig(save_path + '/all_cells_used_for_linReg.svg')

# Get number of cells active vs number of cells inactive

df_all['exp'] = np.nan
exp_id = 0
for fish in df_all.fishlabel.unique():
    for plane in df_all[df_all.fishlabel == fish].plane.unique():
        exp = fish + '_' + plane
        df_all.loc[(df_all.fishlabel == fish) & (df_all.plane == plane), 'exp'] = exp
        df_all.loc[(df_all.fishlabel == fish) & (df_all.plane == plane), 'exp_id'] = exp_id
        exp_id += 1

df_all['active'] = False
for i in df_all.cell_id.unique():
    if df_all[df_all.cell_id == i].recruitment_f.sum() != 0:
        df_all.loc[df_all.cell_id == i, 'active'] = True

all_prop = []
for i, j in enumerate(df_all.exp.unique()):
    print(j)
    if j.split('_')[2] in ["100um", '110um', '120um']:
        print('Not taken here.')
        continue
    n_active = len(df_all[(df_all.active) & (df_all.syl == 0) & (df_all.exp == j)])
    n_all = len(df_all[df_all.exp == j].cell_id.unique())
    all_prop.append(100 * n_active / n_all)

avg_n_active, std_n_active = np.mean(all_prop), np.std(all_prop)
print(avg_n_active, std_n_active)

## Panel g1 and g2: example cells encoded by different features


from scipy.stats import zscore

fish, plane = '210203_F01', '70um_00'
ex_exp = fish + '_' + plane
output_path = summary_csv.loc[(summary_csv.fishlabel == fishlabel) & (summary_csv.plane == plane), 'output_path'].item()
with shelve.open(output_path + '/shelve_calciumAnalysis.out') as f:
    cells = f['cells']
    dff = f['dff_f_lp_inter']
    stat = f['stat']
    fps_ci = f['fps']
regressors = np.load(data_path + ex_exp + '/regressors.npy', allow_pickle=True).item()
ops_path = summary_csv.loc[(summary_csv.fishlabel == fishlabel) & (summary_csv.plane == plane), 'data_path'].item()
ops = np.load(ops_path + '/suite2p/plane0/ops.npy', allow_pickle=True).item()
time_indices_2p = np.arange(dff.shape[1]) / fps_ci

ex_neurons = {'72': ['freq_abs', 'amp_change'], '66': ['freq_abs'], '76': ['amp_abs']}
max_frame = 230

for i, j in enumerate(ex_neurons):
    features = ex_neurons[j]
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(exp + '_cell:' + j)
    ax[0].plot(time_indices_2p[:max_frame], zscore(dff[int(j), :])[:max_frame], label='dff')
    for k in features:
        ax[0].plot(time_indices_2p[:max_frame], zscore(regressors[k])[:max_frame], label=k)
        ax[1].scatter(zscore(dff[int(j), :])[:max_frame], zscore(regressors[k])[:max_frame])
    ax[0].legend()
    ax[1].legend()

#  Find prop of neurons not explained by any regressors

total_forward_neurons = len(df_linReg_all[df_linReg_all.useLinReg].cell_id.unique())

neurons_not_explained = []
for i in features:
    neurons_not_explained.append(list(df_linReg_all[(df_linReg_all.feature_label == i) &
                                                    df_linReg_all.useLinReg &
                                                    ~(df_linReg_all.alpha_norm > 0)].cell_id.unique()))
unique_neurons_not_explained = set(neurons_not_explained[0]).intersection(neurons_not_explained[1],
                                                                          neurons_not_explained[2],
                                                                          neurons_not_explained[3])
print('{}/{} neurons were not explaiend succesfully by the model'.format(len(unique_neurons_not_explained),
                                                                         total_forward_neurons))

# Find prop of neurons explained by the different regressors
dict_prop_neurons_explained = {}
for i in features:
    dict_prop_neurons_explained[i] = df_linReg_all[(df_linReg_all.feature_label == i) &
                                                   df_linReg_all.useLinReg &
                                                   (df_linReg_all.alpha_norm > 0)].cell_id.unique()
    print(i,
          '\n\n{}% neurons explained by {}'.format(len(dict_prop_neurons_explained[i]) * 100 / total_forward_neurons,
                                                   i))
    print('{}/{} neurons explained by {}'.format(len(dict_prop_neurons_explained[i]), total_forward_neurons,
                                                 i))

all_pairs = []
for i in features:
    for j in features:
        if j == i or (j, i) in all_pairs:
            continue

        all_pairs.append((i, j))
        dict_prop_neurons_explained[(i, j)] = list(
            set(dict_prop_neurons_explained[i]).intersection(dict_prop_neurons_explained[j]))

        print('\n\n{}% neurons explained by {} and {}'.format(
            len(dict_prop_neurons_explained[(i, j)]) * 100 / total_forward_neurons,
            i, j))
        print('{}/{} neurons explained by {} and {}'.format(len(dict_prop_neurons_explained[(i, j)]),
                                                            total_forward_neurons,
                                                            i, j))

#  Neurons positevely explained by TBF
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(ops_ref['meanImg'], cmap='Greys')
sns.scatterplot(data=df_linReg_all[df_linReg_all.cell_id.isin(dict_prop_neurons_explained['freq_abs'])],
                x='norm_y_pos', y='norm_x_pos', hue='approx_plane',
                ax=ax
                )
fig.suptitle('Cells positively explained by iTBF regressor, color-code=plane')
fig.savefig(save_path + '/positvely_explained_TBF.svg')

#   Which neurons are explained by both n osc and frequency ?
uniquely_explained_by_freq = []
all_others = df_linReg_all[(df_linReg_all.feature_label.isin(['freq_change', 'amp_abs', 'amp_change'])) &
                           (df_linReg_all.alpha_norm > 0) & df_linReg_all.useLinReg].cell_id.unique()
cell_id_freq_explained = df_linReg_all[(df_linReg_all.feature_label == 'freq_abs') &
                                       (df_linReg_all.alpha_norm > 0) & df_linReg_all.useLinReg].cell_id.unique()
for i in cell_id_freq_explained:
    if not i in all_others:
        uniquely_explained_by_freq.append(i)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(ops_ref['meanImg'], cmap='Greys')
sns.scatterplot(data=df_linReg_all[(df_linReg_all.cell_id.isin(uniquely_explained_by_freq)) &
                                   (df_linReg_all.feature_label == 'freq_abs')],
                x='norm_y_pos', y='norm_x_pos',
                ax=ax
                )
sns.scatterplot(data=df_linReg_all[(df_linReg_all.cell_id.isin(
    set(uniquely_explained_by_freq).intersection(set(succesfull_duration_encoded_cell_id)))) &
                                   (df_linReg_all.feature_label == 'freq_abs')],
                x='norm_y_pos', y='norm_x_pos',
                ax=ax
                )
fig.suptitle('In blue, cells that are explained by iTBF only,\nIn orange, cells that are explained by iTBF only '
             '& are encoding n oscillations')
fig.savefig(save_path + '/uniquely_explained_TBF_and_n_osc_encoding.svg')

# Plot all neurons explained by iTBF and whose max DF/F scales with n osc
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(ops_ref['meanImg'], cmap='Greys')
sns.scatterplot(data=df_linReg_all[(df_linReg_all.cell_id.isin(dict_prop_neurons_explained['freq_abs'])) &
                                   (df_linReg_all.feature_label == 'freq_abs')],
                x='norm_y_pos', y='norm_x_pos',
                ax=ax,
                size='approx_plane',
                )
sns.scatterplot(data=df_linReg_all[(df_linReg_all.cell_id.isin(
    set(dict_prop_neurons_explained['freq_abs']).intersection(set(succesfull_duration_encoded_cell_id)))) &
                                   (df_linReg_all.feature_label == 'freq_abs')],
                x='norm_y_pos', y='norm_x_pos',
                ax=ax,
                size='approx_plane'
                )
fig.suptitle('In blue, cells that are explained by iTBF,\nIn orange, cells that are explained by iTBF'
             '& are encoding n. osc')
fig.savefig(save_path + '/explained_TBF_and_n_osc_encoding.svg')

#  Find cells not well encoding duration and encoding something else
not_duration_encoded = set(df_all[df_all.in_chosen_cluster].cell_id).difference(succesfull_duration_encoded_cell_id)
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(ops_ref['meanImg'], cmap='Greys')
for i in [('freq_abs', 'amp_abs'), ('freq_abs', 'amp_change'), ('amp_abs', 'amp_change')]:
    temp_df = df_linReg_all[df_linReg_all.neuron_id.isin(dict_prop_neurons_explained[i])].drop_duplicates(
        subset='neuron_id')
    ax.scatter(temp_df[temp_df.approx_plane == '90um'].norm_y_pos,
               temp_df[temp_df.approx_plane == '90um'].norm_x_pos, label=i, marker='o')
    ax.scatter(temp_df[temp_df.approx_plane == '70um'].norm_y_pos,
               temp_df[temp_df.approx_plane == '70um'].norm_x_pos, label=i, marker='x')
    if any(temp_df.cell_id.isin(not_duration_encoded)):
        ax.scatter(temp_df[(temp_df.cell_id.isin(not_duration_encoded)) &
                           (temp_df.approx_plane == '90um')].norm_y_pos,
                   temp_df[(temp_df.cell_id.isin(not_duration_encoded)) &
                           (temp_df.approx_plane == '90um')].norm_x_pos,
                   label='not duration encoded', marker='o',
                   facecolors='none', edgecolors='black')
        ax.scatter(temp_df[(temp_df.cell_id.isin(not_duration_encoded)) &
                           (temp_df.approx_plane == '70um')].norm_y_pos,
                   temp_df[(temp_df.cell_id.isin(not_duration_encoded)) &
                           (temp_df.approx_plane == '70um')].norm_x_pos,
                   label='not duration encoded', marker='X',
                   facecolors='none', edgecolors='black')
ax.legend()

fig.savefig(save_path + '/double_explained_TBF_and_NOT_duration_encoding.svg')

# NOT SAVED HERE IN JAN
df_all.to_csv(
    '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/V2a_recruitment_behavior/analysis_11/df_with_linReg.csv')

df_all.to_pickle(
    '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/V2a_recruitment_behavior/analysis_11/df_with_linReg.pkl')

# ONLY SAVED HERE IN JAN

df_all.to_pickle(save_path +
                 '/df_with_linReg.pkl')

df_all.to_csv(save_path +
              '/df_with_linReg.csv')

df_linReg_all.to_pickle(save_path +
                        '/df_linReg.pkl')

df_linReg_all.to_csv(save_path +
                     '/df_linReg.csv')
