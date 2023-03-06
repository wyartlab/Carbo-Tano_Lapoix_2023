import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
from utils.import_data import load_suite2p_outputs
from utils.calcium_traces import get_pos_x, get_pos_y, plot_cells_side, plot_cells_group
from utils.ref_pos import addRefPos


output_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/dataset_final/'
df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/data_summary_II.csv', )

dict_all = {}
cell_id = 0
for i in df_summary.index:

    df_lf = pd.read_pickle(df_summary.output_path[i] + '/dataset/df_lf')

    for col in ['fishlabel', 'plane', 'real_plane', 'dv_group']:
        if col not in df_lf:
            df_lf[col] = df_summary[col][i]
    del col

    for cell in set(df_lf.cell):
        df_lf.loc[df_lf.cell == cell, 'cell_id'] = cell_id
        cell_id += 1
    del cell

    if 'norm_x_pos' not in df_lf:
        x_ref, y_ref = (435, 265)
        df_lf = addRefPos(df_summary.fishlabel[i], df_summary, x_ref, y_ref, df_lf)

    dict_all[str(i)] = df_lf

df_fish_all = pd.concat(dict_all, ignore_index=True)
del dict_all
del df_lf

df_fish_all.to_pickle(output_path + '/df_paralysed_preprocess.pkl')
df_fish_all.to_csv(output_path + '/df_paralysed_preprocess.csv')

# Add hidden cells

df_fish_all['detected'] = 1
dict_added = {}
for i in df_summary.index:
    plane = df_summary.plane[i]
    data_path = df_summary.data_path[i]
    df_lf = pd.read_pickle(df_summary.output_path[i] + '/dataset/df_lf')

    _, _, _, old_stat, old_ops, old_iscell = load_suite2p_outputs(plane, data_path + '/')
    _, _, _, new_stat, new_ops, new_iscell = load_suite2p_outputs(plane,
                                                                  data_path + '/Total_cell_number/')
    old_cells = np.flatnonzero(old_iscell[:, 0])
    new_cells = np.flatnonzero(new_iscell[:, 0])
    del old_iscell

    if len(old_cells) != len(new_cells):

        print('\n\n{}, {}\nNew cells added !\n'.format(df_summary.fishlabel[i], df_summary.plane[i]))

        old_cells_pos = [old_stat[cell]['med'] for cell in old_cells]

        added = []
        for cell in new_cells:
            if new_stat[cell]['med'] not in old_cells_pos:
                added.append(cell)
        del old_cells_pos
        del cell

        print('\n', i, '_nMatching old + new cells :', len(new_cells) - len(old_cells) == len(added))
        print('{} new cells detected, \n{}% of total cells were manually added'.format(len(added),
                                                                                       round(len(added) * 100 / len(
                                                                                           new_cells), 2)))

        df_added = pd.DataFrame(index=range(len(added) * len(set(df_lf.stim_intensity))),
                                columns=df_fish_all.columns)

        for col in ['fishlabel', 'plane', 'real_plane', 'dv_group']:
            df_added[col] = df_summary[col][i]

        df_added['stim_intensity'] = np.tile(list(set(df_lf.stim_intensity)), len(added))
        df_added['stim'] = np.tile(list(set(df_lf.stim)), len(added))
        df_added['cell'] = np.repeat([1000 + k for k in range(len(added))], len(set(df_lf.stim)))

        # Get group and pos

        try:
            bl_input = df_summary['bulbar_lat_new'][i].split(',')
            bulbar_lateral = list(map(int, bl_input))
        except AttributeError:
            print('No bulbar lateral cells found from user, or mistyped in the csv file.')
            bulbar_lateral = list()  # bulbar lateral will be empty

        for j, cell in enumerate(added):
            # get cell group by user input
            # if cell in bulbar lateral
            if cell in bulbar_lateral:
                df_added.loc[df_added.cell == j + 1000, 'group'] = 'bulbar_lateral'
            # else, get cell gorup by x position vis à vis of x limits defined by user
            elif get_pos_x(cell, new_stat) > df_summary['sc_bulbar'][i]:
                df_added.loc[df_added.cell == j + 1000, 'group'] = 'spinal_cord'
            elif df_summary['bulbar_pontine'][i] < get_pos_x(cell, new_stat) <= df_summary[
                'sc_bulbar'][i]:
                df_added.loc[df_added.cell == j + 1000, 'group'] = 'bulbar_medial'
            else:
                df_added.loc[df_added.cell == j + 1000, 'group'] = 'pontine'

            # now define the side of the cell
            if get_pos_y(cell, new_stat) < df_summary['midline'][i]:
                df_added.loc[df_added.cell == j + 1000, 'side'] = 'ipsi'
            else:
                df_added.loc[df_added.cell == j + 1000, 'side'] = 'contra'

            df_added.loc[df_added.cell == j + 1000, 'x_pos'] = get_pos_x(cell, new_stat)
            df_added.loc[df_added.cell == j + 1000, 'y_pos'] = get_pos_y(cell, new_stat)

            # Adding cell id
            df_added.loc[df_added.cell == j + 1000, 'cell_id'] = cell_id
            cell_id += 1

        df_added['detected'] = 0

        x_ref, y_ref = (435, 265)
        df_added_final = addRefPos(df_summary.fishlabel[i], df_summary, x_ref, y_ref, df_added)
        dict_added[str(i)] = df_added_final
        del df_added
        del df_added_final

    del plane
    del data_path
    del df_lf

df_all = df_fish_all.append(pd.concat(dict_added, ignore_index=True), ignore_index=True)
del dict_added

df_all.to_pickle(output_path + '/df_paralysed_all_cells.pkl')

#  Select cells of interest

df_all['recruitment_state'] = ['not active'] * len(df_all)
df_all['stim_intensity'] = df_all['stim_intensity'].astype('float')

# remove high stim intensities, we won't take them into account, and also the .5 that are only in one fish
stim_intensities = [1, 2, 3, 4, 5]  #  stim int to keep
df = df_all[df_all.stim_intensity.isin(stim_intensities)].copy()
df['picked'] = 0
df['first_response'] = np.nan
df['first_response_amp'] = np.nan
df['second_response'] = np.nan
df['response_num'] = np.nan
df['second_response_amp'] = np.nan

# Select cells of interest (once they start responding, always respond

for cell in set(df[df.detected == 1].cell_id):

    recruitment = list(df.loc[df.cell_id == cell, 'recruitment_f'])
    stim_intensities = list(df.loc[df.cell_id == cell, 'stim_intensity'])

    a = next((i for i in range(len(recruitment)) if recruitment[i] == 1), None)  # first stim where recruited

    if (a is not None) & (len(set(recruitment[a:])) == 1):
        # if recruitment was found and a is not in the highest stim ints

        df.loc[df.cell_id == cell, 'picked'] = 1

        first_int = int(stim_intensities[a])

        # fill threshold info for this cell
        df.loc[df.cell_id == cell, 'first_response'] = first_int
        first_amp = float(df.loc[(df.stim_intensity == first_int) & (df.cell_id == cell), 'max_dff_f'])
        df.loc[df.cell_id == cell, 'first_response_amp'] = first_amp
        df.loc[(df.stim_intensity == first_int) & (df.cell_id == cell),
               'response_num'] = 'first_response'
        second_response = int(first_int + 1)
        # if any(df.stim_intensity == second_response):
        second_response_amp = df.loc[(df.cell_id == cell) &
                                     (df.stim_intensity == second_response), 'max_dff_f']
        df.loc[df.cell_id == cell, 'second_response_amp'] = second_response_amp
        df.loc[(df.cell_id == cell) &
               (df.stim_intensity == second_response), 'response_num'] = 'second_response'

        #  recruitment state
        df.loc[(df.stim_intensity == first_int) & (df.cell_id == cell),
               'recruitment_state'] = 'newly_active'

        for stim_int in range(first_int + 1, 6):
            df.loc[(df.stim_intensity == stim_int) & (df.cell_id == cell),
                   'recruitment_state'] = 'already_active'

print('Selected {} out of {} cells'.format(len(set(df[df.picked == 1].cell_id)), len(set(df.cell_id))))

#  Compute ratio

df['ratio_second_first_resp'] = df['second_response_amp'] / df['first_response_amp']

# fill info for not detected / not selected cells
df.loc[df.picked == 0, 'first_response'] = 'not_selected'
df.loc[df.picked == 0, 'recruitment_state'] = 'not_selected'
df.loc[df.detected == 0, 'first_response'] = 'not_detected'
df.loc[df.detected == 0, 'recruitment_state'] = 'not_detected'

# add if cell was ever recruited or not

df['category'] = np.nan
for cell in set(df.cell_id):
    if df[df.cell_id == cell].first_response.iloc[0] == 'not_selected':
        if any(i == 1 for i in df[df.cell_id == cell].recruitment_f):
            df.at[df.cell_id == cell, 'category'] = 'recruited_not_selected'
        else:
            df.at[df.cell_id == cell, 'category'] = 'not_recruited'
    elif df[df.cell_id == cell].first_response.iloc[0] in([1.0, 2.0, 3.0, 4.0, 5.0]):
        df.at[df.cell_id == cell, 'category'] = 'reliably_recruited'
    else:
        df.at[df.cell_id == cell, 'category'] = 'not_recruited'

#  ADD PARAM OF IF CELL IS RETICULOSPINAL OR NOT

df['RSN'] = [df.norm_x_pos[i] >= -100 for i in df.index]


#  Assign birthdate


def assign_brithdate(cell, df):
    if list(df.loc[df.cell_id == cell, 'group'])[0] == 'pontine':
        if list(df.loc[df.cell_id == cell, 'real_plane'])[0] <= 223:
            output = 'early-born'
        else:
            output = 'late-born'
    elif list(df.loc[df.cell_id == cell, 'group'])[0] == 'bulbar_lateral':
        output = 'early-born'
    elif list(df.loc[df.cell_id == cell, 'group'])[0] == 'bulbar_medial':
        output = 'late-born'
    else:
        output = np.nan
    return output


def assign_final_cell_group(cell, df):
    x = list(df.loc[df.cell_id == cell, 'norm_x_pos'])[0]
    plane = list(df.loc[df.cell_id == cell, 'real_plane'])[0]
    group = list(df.loc[df.cell_id == cell, 'group'])[0]
    if x <= -60:
        output = 'prepontine'
    elif (-60 < x <= 50) & (plane <= 220):
        output = 'pontine_ventral'
    elif (-60 < x <= 50) & (plane > 220):
        output = 'pontine_dorsal'
    elif (50 < x <= 148) & (plane <= 220):
        output = 'retropontine_ventral'
    elif (50 < x <= 148) & (plane > 220):
        output = 'retropontine_dorsal'
    else:
        if group == 'spinal_cord':
            output = 'spinal_cord'
        elif group == 'bulbar_lateral':
            output = 'medullar_lateral'
        else:
            output = 'medulla'

    # if list(df.loc[df.cell_id == cell, 'group'])[0] == 'pontine':
    #     if list(df.loc[df.cell_id == cell, 'birthdate'])[0] == 'early-born':
    #         output = 'pontine_ventral'
    #     else:
    #         output = 'pontine_dorsal'
    # else:
    #     output = list(df.loc[df.cell_id == cell, 'group'])[0]

    return output


background = np.load('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/Calcium_Imaging//'
                     '201007_F02_chx10/allPlanes/background.npy')

fig, ax = plt.subplots()
ax.imshow(background, cmap='Greys', vmax=400)

for cell in set(df.cell_id):
    df.loc[df.cell_id == cell, 'birthdate'] = assign_brithdate(cell, df)
    df.loc[df.cell_id == cell, 'final_cell_group'] = assign_final_cell_group(cell, df)

df.loc[df.detected == 0, 'recruitment_f'] = 'not_detected'

# Compute median and mean values of kinematics parameters

for cell in set(df[df.slope_rise.notna()].cell_id):
    df.at[df.cell_id == cell, 'median_slope_rise'] = df[df.cell_id == cell].slope_rise.median()
    df.at[df.cell_id == cell, 'mean_slope_rise'] = df[df.cell_id == cell].slope_rise.mean()
    df.at[df.cell_id == cell, 'median_time_rise'] = df[df.cell_id == cell].time_rise.median()
    df.at[df.cell_id == cell, 'mean_time_rise'] = df[df.cell_id == cell].time_rise.mean()

df['median_slope_rise'] = df['median_slope_rise'].astype('float64')
df['mean_slope_rise'] = df['mean_slope_rise'].astype('float64')
df['median_time_rise'] = df['median_time_rise'].astype('float64')
df['mean_time_rise'] = df['mean_time_rise'].astype('float64')
df['slope_rise'] = df['slope_rise'].astype('float64')

# Df proportions

df_prop = pd.DataFrame(columns=list(set(df.category)),
                       index=list(set(df.final_cell_group)))


for group in df_prop.index:
    total = len(set(df[(df.final_cell_group == group)].cell_id))
    for cat in df_prop.columns:
        df_prop.loc[group, cat] = 100 * len(
            set(df[(df.final_cell_group == group) & (df.category == cat)].cell_id)) / total

all_fish = {}
for fish in set(df.fishlabel):
    df_prop_fish = pd.DataFrame(columns=list(set(df.category)),
                                index=list(set(df.final_cell_group)))
    for group in df_prop_fish.index:
        total = len(set(df[(df.final_cell_group == group) & (df.fishlabel == fish)].cell_id))
        for cat in df_prop_fish.columns:
            try:
                df_prop_fish.loc[group, cat] = 100 * len(
                    set(df[(df.final_cell_group == group) & (df.category == cat) & (df.fishlabel == fish)].cell_id)) / total
            except ZeroDivisionError:
                continue
    fig, ax = plt.subplots()
    fig.suptitle(fish)
    df_prop_fish.plot.bar(ax=ax)


    df_prop_fish['fishlabel'] = fish
    all_fish[fish] = df_prop_fish

df_prop_all = pd.concat(all_fish)
df_prop_all.to_csv(output_path + '/df_prop_per_fish.csv')

# Final saving

df.to_pickle(output_path + '/df_final.pkl')
df.to_csv(output_path + '/df_final.csv')
df_prop.to_pickle(output_path + '/df_final_prop.pkl')
df_prop.to_csv(output_path + '/df_final_prop.csv')

# Proportion of reliably recruited cells, per fish, per group and per stim

dict_prop_recruited = {}

for fish in set(df.fishlabel):
    for group in set(df.final_cell_group):
        df_temp = pd.DataFrame(columns=['stim','fish','group','prop_reliably_recruited_cells', 'prop_recruited_cells'],
                               index=[1,2,3,4,5])
        df_temp['fish'] = fish
        df_temp['group'] = group
        df_temp['stim'] = [1,2,3,4,5]
        nCells = len(set(df[(df.fishlabel == fish) & (df.final_cell_group == group)].cell_id))
        for stim in df_temp.index:
            n_reliably_r = len(df[(df.fishlabel == fish) & (df.final_cell_group == group)
                                       & (df.stim_intensity == stim) & (df.recruitment_f == 1)
                                       & (df.picked == 1)])
            n_recruited = len(df[(df.fishlabel == fish) & (df.final_cell_group == group)
                                       & (df.stim_intensity == stim) & (df.recruitment_f == 1)])

            try:
                df_temp.loc[stim, 'prop_reliably_recruited_cells'] = 100*n_reliably_r/nCells
                df_temp.loc[stim, 'prop_recruited_cells'] = 100*n_recruited/nCells
            except ZeroDivisionError:
                continue

        dict_prop_recruited[fish+'_'+group] = df_temp

df_prop_recruited = pd.concat(dict_prop_recruited)
df_prop_recruited.to_csv(output_path + '/df_prop_recruited.csv')

# PLOTS

