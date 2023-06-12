import pandas as pd
import numpy as np

# For paper Carbo-Tano, Lapoix 2023
# From dataset: MLR electrical stimulation, different params, while recording tail angle of the zebrafish larvae
# Script function: for each stimulation sites, outputs electrode's position and summary of behavior elicited

df_electrode = pd.read_csv('./df_electrode_placement.csv')
df_behavior = pd.read_csv('./df.csv')
df_bout = pd.read_csv('./new_df_bout_all.csv')

for stim_id, df_index in enumerate(df_behavior[df_behavior.condition == 'stim'].index):

    df_behavior.at[df_index, 'stim_id'] = stim_id

    if df_behavior.n_bouts[df_index] == 0:
        output = np.nan
        
    # Compute ratio of forward vs struggle elicitied at this stim site, for this trial
    else: 
        if df_behavior.stim_int[df_index] == 0.1:  #  for 40s stim duration
            print('40s stim, taking proportion of time spent swimming')
            fish, trial = df_behavior.fishlabel[df_index], df_behavior.trial[df_index]
            n_bouts = len(df_bout[(df_bout.fishlabel == fish) & (df_bout.trial == trial) &
                                  (df_bout.bout_condition == 'during_stim') & (df_bout.start_time < 25)])
            n_bouts_f = len(df_bout[(df_bout.fishlabel == fish) & (df_bout.trial == trial) &
                                    (df_bout.bout_condition == 'during_stim') & (df_bout.start_time < 25) &
                                    (df_bout.manual_cat == 'F')])
            n_bouts_s = len(df_bout[(df_bout.fishlabel == fish) & (df_bout.trial == trial) &
                                    (df_bout.bout_condition == 'during_stim') & (df_bout.start_time < 25) &
                                    (df_bout.manual_cat == 'S')])
        else:
            n_bouts = df_behavior.loc[df_index, 'n_bouts']
            n_bouts_f = df_behavior.loc[df_index, 'n_bouts_f']
            n_bouts_s = df_behavior.loc[df_index, 'n_bouts_s']

        if n_bouts_f == 0:
            df_behavior.at[df_index, 'ratio_f_s'] = -1
        elif n_bouts_s == 0:
            df_behavior.at[df_index, 'ratio_f_s'] = 1
        else:
            df_behavior.at[df_index, 'ratio_f_s'] = (n_bouts_f - n_bouts_s) / n_bouts

    # Get distance from center of the electrode placement

    electrode_placement = df_behavior.loc[df_index, 'electrode_placement']
    try:
        distance = float(df_electrode.loc[
                             df_electrode.electrode_placement == electrode_placement, 'distance_to_center_of_stim'])
        x_elec, y_elec, z_elec = float(df_electrode.loc[df_electrode.electrode_placement == electrode_placement, 'x']), \
                                 float(df_electrode.loc[df_electrode.electrode_placement == electrode_placement, 'y']), \
                                 float(df_electrode.loc[df_electrode.electrode_placement == electrode_placement, 'z'])
        median_ratio = float(df_electrode.loc[df_electrode.electrode_placement == electrode_placement,
                                              'median_ratio_f_s'])
        print('\n\nelectrode placement', electrode_placement,
              '\n stim id:', stim_id,
              '\nratio F/S for this stim:', df_behavior.loc[df_index, 'ratio_f_s'],
              '\nmean ratio F/S:', median_ratio)
    except TypeError:
        distance = np.nan
        x_elec, y_elec, z_elec = np.nan, np.nan, np.nan
        median_ratio = np.nan

    df_behavior.at[df_index, 'electrode_distance_to_center'] = distance
    df_behavior.at[df_index, 'x_electrode'] = x_elec
    df_behavior.at[df_index, 'y_electrode'] = y_elec
    df_behavior.at[df_index, 'z_electrode'] = z_elec
    df_behavior.at[df_index, 'mean_ratio_f_s'] = median_ratio

df_behavior.to_csv('./df_per_stim.csv')

