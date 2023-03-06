#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:45:23 2022

@author: mathildelpx
"""
#%% Init

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multipagetiff as mtif
from glob import glob
import seaborn as sns


# %% Load data

in_path = './'

total_cell = 0
for i, path in enumerate(glob(in_path+'/*.csv')):
    filename = os.path.basename(path)
    df_fish = pd.read_csv(path)
    fishlabel = filename.split('_')[0]+'_'+filename.split('_')[1]
    df_fish['fishlabel'] = fishlabel
    df_fish['trial_ID'] = filename.split('_')[2]+'_'+filename.split('_')[3]
    df_fish['unique_cell_id'] = df_fish['cell_id']+total_cell
    if fishlabel != '220119_F2':
        temp_x = np.array(df_fish['reg_z_pos'].copy())
        temp_z= np.array(df_fish['reg_x_pos'].copy())
        df_fish['reg_z_pos'] = temp_z
        df_fish['reg_x_pos'] = temp_x

    # TODO: remove once the registration was properly done
    if fishlabel == '220210_F1':
        df_fish['reg_y_pos'] += 25
    total_cell += len(df_fish.cell_id.unique())
    
    if i == 0:
        df_all = df_fish
    else:
        df_all = pd.concat([df_all, df_fish],
                           ignore_index=True)
        
ref_background = mtif.read_stack(os.path.join(in_path, 
                                              'ref_threshold.tif'))

array_background = ref_background.pages

# %% Plot

df_shorten = df_all.drop_duplicates('unique_cell_id')


fig, ax = plt.subplots()
ax.imshow(array_background.max(axis=0), 
          vmax=100,
          cmap='Greys')
sns.scatterplot(data=df_shorten,
                x='reg_x_pos',
                y='reg_y_pos',
                hue='fishlabel',
                ax=ax)

fig, ax = plt.subplots()
ax.imshow(array_background.max(axis=1), 
          vmax=100,
          cmap='Greys')
sns.scatterplot(data=df_shorten,
                x='reg_x_pos',
                y='reg_z_pos',
                hue='fishlabel',
                ax=ax)

ax.set_ylim(170, -20)
ax.set_xlim(0, 500)

# %% set function to map


def plot_all_views_density(temp_df, 
                           bout_type,
                           vmax):
    # dorsal view
    x='reg_x_pos'
    y='reg_y_pos'
    sns.set_palette('magma')
    sns.displot(temp_df,
                x=x,
                y=y,
                height=3.3,
                aspect=2.8,
                vmin=0,
                vmax=vmax)
    
    plt.title('Cells more active during  {}: dorsal view'.format(bout_type))
    plt.xlim(0,500)
    plt.ylim(250,120)
    plt.savefig('./vmax_{}/{}_heatmap_dorsal.svg'.format(vmax, bout_type))

    plt.savefig('./vmax_{}/{}_heatmap_dorsal.png'.format(vmax, bout_type))

    # sagittal view
    x='reg_x_pos'
    y='reg_z_pos'
    sns.set_palette('magma')
    sns.displot(temp_df,
                x=x,
                y=y,
                height=3,
                aspect=2.5,
                vmin=0,
                vmax=vmax)
    
    plt.title('Cells more active during  {}: sagittal view'.format(bout_type))
    plt.xlim(0, 500)
    plt.ylim(170,-20)
    plt.savefig('./vmax_{}/{}_heatmap_sagittal.svg'.format(vmax, bout_type))

    plt.savefig('./vmax_{}/{}_heatmap_sagittal.png'.format(vmax, bout_type))

    # axial view, in 2 sections (rostral and caudal)

vmax = 30

temp_df = df_all[(df_all.bout_type == 'forward') & 
                     (df_all.more_active == True) &
                     ~(df_all.fishlabel == '220210_F1')]
    
plot_all_views_density(temp_df, bout_type='forward',vmax=vmax)
    
        
temp_df = df_all[(df_all.bout_type == 'left_turns') & 
                     (df_all.more_active == True) &
                     ~(df_all.fishlabel == '220210_F1')]
    
plot_all_views_density(temp_df, bout_type='left_turns',vmax=vmax)
    
temp_df = df_all[(df_all.bout_type == 'right_turns') & 
                     (df_all.more_active == True) &
                     ~(df_all.fishlabel == '220210_F1')]
    
plot_all_views_density(temp_df, bout_type='right_turns',vmax=vmax)
temp_df = df_all[(df_all.forward_component) & ~(df_all.fishlabel == '220210_F1')].drop_duplicates('unique_cell_id')
    
plot_all_views_density(temp_df, 
                                           bout_type='forward_component',vmax=vmax)

# %% FORWARD
    
temp_df = df_all[(df_all.bout_type == 'forward') & 
                 (df_all.more_active == True) &
                 ~(df_all.fishlabel == '220210_F1')]


# fig_forward = plot_all_views(temp_df, bout_type='forward')
# fig_forward.savefig('fig_forward.svg')
plot_all_views_density(temp_df, bout_type='forward')
#fig_density_forward.savefig('fig_density_dist_forward.svg')

# %% LEFT TURNS
    
temp_df = df_all[(df_all.bout_type == 'left_turns') & 
                 (df_all.more_active == True) &
                 ~(df_all.fishlabel == '220210_F1')]

plot_all_views_density(temp_df, bout_type='left_turns')



# %% RIGHT TURNS
# normalize vmax and vmin for all
temp_df = df_all[(df_all.bout_type == 'right_turns') & 
                 (df_all.more_active == True) &
                 ~(df_all.fishlabel == '220210_F1')]

plot_all_views_density(temp_df, bout_type='right_turns')

# %% FORWARD COMPONENT
temp_df = df_all[(df_all.forward_component) & ~(df_all.fishlabel == '220210_F1')].drop_duplicates('unique_cell_id')

plot_all_views_density(temp_df, 
                                       bout_type='forward_component')
# %% STats

for fish in fishes:
    print(fish)
    df_recruitment = pd.read_pickle(glob(in_path+'/df_recruitment/{}*.pkl'.format(fish))[0])
    n_bouts = len(df_recruitment.bout_id.unique())
    MASKS = {'forward': df_recruitment.abs_Max_Bend_Amp < 25,
         'left_turns': (df_recruitment.abs_Max_Bend_Amp >= 25) & (df_recruitment.abs_Max_Bend_Amp < 60) & (
                 df_recruitment.mean_tail_angle > 0),
         'right_turns': (df_recruitment.abs_Max_Bend_Amp >= 25) & (df_recruitment.abs_Max_Bend_Amp < 60) & (
                 df_recruitment.mean_tail_angle < 0)}
    for bout_type in df_all.bout_type.unique():
        n_bout_type = len(df_recruitment[MASKS[bout_type]].bout_id.unique())
        print('{}: {}/{}'.format(bout_type,
                                 n_bout_type,
                                 n_bouts))

# %% STats 2
fishes = df_all[(df_all.bout_type == 'left_turns') & 
                 (df_all.more_active == True) &
                 ~(df_all.fishlabel == '220210_F1')].fishlabel.unique().copy()
all_temp = {}
all_temp['left_steering'] = df_all[(df_all.bout_type == 'left_turns') & 
                 (df_all.more_active == True) &
                 ~(df_all.fishlabel == '220210_F1')]
all_temp['right_steering'] = df_all[(df_all.bout_type == 'right_turns') & 
                 (df_all.more_active == True) &
                 ~(df_all.fishlabel == '220210_F1')]
all_temp['forward_component'] = df_all[(df_all.forward_component) & ~(df_all.fishlabel == '220210_F1')].drop_duplicates('unique_cell_id')

props = {'left_steering': [],
         'right_steering': [],
         'forward_component': []}

for fish in fishes:
    print('\n', fish)
    total_n_cells = len(df_all[df_all.fishlabel == fish].cell_id.unique())
    for key in all_temp.keys():
        recruited_cells = len(all_temp[key][all_temp[key].fishlabel == fish].cell_id.unique())
        print('Number of cells recruited for {}: {}'.format(key,
                                                            str(recruited_cells)))
        
        print('Total number of cells recruited for {}: {}'.format(key,
                                                            str(total_n_cells)))
        print('Prop of cells recruited for {}: {}'.format(key,
                                                          str(recruited_cells/total_n_cells)))
        props[key].append(recruited_cells/total_n_cells)

for key in props.keys():
    print('\n')
    print(key)
    print(props[key])
    print('median', np.median(props[key]))
    print('std', np.std(props[key]))
# %% Heatmaps

temp_df = df_all[(df_all.bout_type == 'right_turns') & 
                 ~(df_all.fishlabel == '220210_F1')]

x='reg_x_pos'
y='reg_y_pos'
sns.displot(temp_df,
            x=x,
            y=y,
            height=3.3,
            aspect=2.8,
            vmin=0,
            vmax=100)
plt.colorbar()

# %% Save

df_all.to_csv('df_all.csv')
df_all.to_pickle('df_all.pkl')



