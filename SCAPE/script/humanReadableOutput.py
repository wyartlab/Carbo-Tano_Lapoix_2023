import numpy as np
import os
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/home/mathilde.lapoix/PycharmProjects/SCAPE/')

import utils.import_data as load

# %% Initialise


def fix_lustre_name(df):
    return df.replace('iss01', 'iss02', regex=True)


def fix_lustre_name_Exp(Exp):
    setattr(Exp, 'fig_path', Exp.fig_path.replace('iss01', 'iss02'))
    setattr(Exp, 'savePath', Exp.savePath.replace('iss01', 'iss02'))


summary_csv = fix_lustre_name(load.load_summary_csv(
    'https://docs.google.com/spreadsheets/d/1VHFmX8j8rfwDiKghT5tb0LxZfmB5_qrgqYmcJxmB7RE/edit#gid=1097839266'))
exp_id = 58

with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Exp.pkl', 'rb') as f:
    Exp = pickle.load(f)
print(Exp.savePath, Exp.runID)

fix_lustre_name_Exp(Exp)
df_frame, df_bout = load.load_behavior_dataframe(Exp)

with open(summary_csv.savePath[exp_id] + summary_csv.run[exp_id] + '/Cells.pkl', 'rb') as f:
    Cells = pickle.load(f)

# Build output signal

cells_fluorescence_signals = np.zeros((len(Cells), Exp.nFramesSCAPE))
cells_spike_rate_signals = np.zeros((len(Cells), Exp.nFramesSCAPE))
cells_positions = np.zeros((len(Cells), 3))

for i, Cell in enumerate(Cells):
    cells_fluorescence_signals[i] = Cell.F_corrected
    cells_spike_rate_signals[i] = Cell.spks
    cells_positions[i,0] = Cell.x_pos
    cells_positions[i,1] = Cell.norm_y_pos
    cells_positions[i,2] = Cell.plane
#
# for i, Cell in enumerate(Cells):
#     if Cell.norm_y_pos < 100 or Cell.norm_y_pos > 200:
#         print('excluding cell:', i)
#         continue
#     cells_fluorescence_signals[i] = Cell.F_corrected
#     cells_spike_rate_signals[i] = Cell.spks
#     cells_positions[i,0] = Cell.x_pos
#     cells_positions[i,1] = Cell.norm_y_pos
#     cells_positions[i,2] = Cell.plane
#
#
# def exclude_empty_rows(arr):
#     return arr[arr.max(axis=1) != 0]
#
#
# cells_positions = exclude_empty_rows(cells_positions)
# cells_spike_rate_signals = exclude_empty_rows(cells_spike_rate_signals)
# cells_fluorescence_signals = exclude_empty_rows(cells_fluorescence_signals)
 # %% Save
 
np.save(os.path.join(Exp.savePath,
                     Exp.runID,
                     '{}_{}_cells_fluorescence_signals.npy'.format(Exp.fishID, Exp.runID)),
        cells_fluorescence_signals)
np.save(os.path.join(Exp.savePath,
                     Exp.runID,
                     '{}_{}_cells_spike_rate_signals.npy'.format(Exp.fishID, Exp.runID)),
        cells_spike_rate_signals)
np.save(os.path.join(Exp.savePath,
                     Exp.runID,
                     '{}_{}_cells_positions.npy'.format(Exp.fishID, Exp.runID)),
        cells_positions)
out_dict = { your_key: Exp.__getattribute__(your_key) for your_key in ['fishID', 'runID', 'stage',
                                                                       'date', 'enucleated',
                                                                       'nFramesSCAPE', 'bad_frames', 'frameRateSCAPE',
                                                                       'frameRateBeh']}

out_dict['nCells'] = cells_positions.shape[0]
print(out_dict)


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


with open(os.path.join(Exp.savePath, Exp.runID, 'analysis_info.json'), 'w') as f:
    json.dump(out_dict, f, default=np_encoder)

# mean background
np.save(os.path.join(Exp.savePath, Exp.runID, 'mean_background.npy'), Exp.mean_background.copy())

# %% About recruitment during bout

# If not done, save output dataframe from df recruitment
df_recruitment = pd.read_pickle(os.path.join(Exp.savePath, Exp.runID, 'df_recruitment.pkl'))

# for 220210_F1_run6, load shorten dataframe (from time 100s only)
# df_recruitment = pd.read_pickle(os.path.join(Exp.savePath, Exp.runID, 'df_recruitment_shorten.pkl'))

if (Exp.fishID == '220119_F2') & (Exp.runID == 'F2_run11'):
    cells_positions_reg = pd.DataFrame(data=cells_positions,
                                       columns=['x', 'y', 'z'])
else:
    cells_positions_reg = pd.read_csv(os.path.join(Exp.savePath, Exp.runID,
                                                   '{}_{}_reg_cells_pos.csv'.format(Exp.fishID.split('_')[0], Exp.runID)))

# Build output dataframe
df_cell_unique = df_recruitment.drop_duplicates('cell_id')
df = pd.DataFrame({'cell_id': np.repeat(df_cell_unique.cell_id, 3),
                   'cell_num': np.repeat(df_cell_unique.cell_num, 3),
                   'x_pos': np.repeat(df_cell_unique.x_pos, 3),
                   'y_pos': np.repeat(df_cell_unique.y_pos, 3),
                   'norm_y_pos': np.repeat(df_cell_unique.norm_y_pos, 3),
                   'reg_x_pos': np.repeat(list(cells_positions_reg.loc[:,'x']), 3),
                   'reg_y_pos': np.repeat(list(cells_positions_reg.loc[:,'y']), 3),
                   'reg_z_pos': np.repeat(list(cells_positions_reg.loc[:,'z']), 3),
                   'side': np.repeat(df_cell_unique.side, 3),
                   'plane': np.repeat(df_cell_unique.plane, 3),
                   'group': np.repeat(df_cell_unique.group, 3),
                   'bout_type': np.tile(['forward', 'left_turns', 'right_turns'], len(df_cell_unique)),
                   'more_active': np.nan,
                   'forward_component': np.nan,
                   'left_only': np.nan,
                   'right_only': np.nan})


def get_if_more_active(id, df, df_recruitment):

    cell_id = df.loc[id, 'cell_id']
    bout_type = df.loc[id, 'bout_type']

    mask_df_recruitment = df_recruitment.cell_id == cell_id

    output = df_recruitment.loc[mask_df_recruitment, 'more_active_during_bout_type_{}'.format(bout_type)].unique()[0]

    return output


def cell_forward_component(cell_id, df):
    mask = df.cell_id == cell_id
    if all(df.loc[mask, 'more_active']):
        return True
    else:
        return False


def cell_left_only(cell_id, df):

    df_cell = df[df.cell_id == cell_id]

    if list(df_cell.loc[df_cell.bout_type == 'left_turns', 'more_active'])[0] is True:
        if any(df_cell.loc[df_cell.bout_type != 'left_turns', 'more_active']):
            return False
        else:
            return True

    else:
        return False


def cell_right_only(cell_id, df):

    df_cell = df[df.cell_id == cell_id]

    if list(df_cell.loc[df_cell.bout_type == 'right_turns', 'more_active'])[0] is True:
        if any(df_cell.loc[df_cell.bout_type != 'right_turns', 'more_active']):
            return False
        else:
            return True

    else:
        return False


df = df.reset_index()
df['more_active'] = pd.Series(df.index).apply(get_if_more_active, args=(df, df_recruitment))
df['forward_component'] = pd.Series(df.cell_id).apply(cell_forward_component, args=(df,))
df['left_only'] = pd.Series(df.cell_id).apply(cell_left_only, args=(df,))
df['right_only'] = pd.Series(df.cell_id).apply(cell_right_only, args=(df,))

df.to_csv(os.path.join(Exp.savePath, Exp.runID, '{}_{}_df_per_cell.csv'.format(Exp.fishID, Exp.runID)))

# Get list of cells possibly emitter or receiver, in the context of generation of forward locomotion
# mask of cells based on position
# as of 8/11/22, reg_x and reg_z are inverted in non reference fish !!!! (was inverted to compute emitter and receiver cells)
receiver_mask = (df.reg_x_pos > 160) & (df.reg_x_pos < 340)
emitter_mask = (df.reg_x_pos > 340) & (df.reg_x_pos < 460) & (df.reg_z_pos > 50)

# mask of cells based on activity
df_receiver = df[(df.more_active == True) & (df.bout_type == 'forward') & receiver_mask]
df_emitter= df[(df.more_active == True) & (df.bout_type == 'forward') & emitter_mask]
a = df.drop_duplicates('cell_id')

fig, ax = plt.subplots(2,1)
fig.suptitle('Emitter (blue) and receiver (red) cells (based on position and activity during forward)')
ax[0].imshow(Exp.mean_background, cmap='Greys')
ax[0].plot(a.reg_x_pos, a.reg_y_pos, 'o', color='grey')
ax[0].plot(df_receiver.reg_x_pos, df_receiver.reg_y_pos, 'ro')
ax[0].plot(df_emitter.reg_x_pos, df_emitter.reg_y_pos, 'bo')
ax[1].plot(a.reg_x_pos, a.reg_z_pos, 'o', color='grey')
ax[1].plot(df_receiver.reg_x_pos, df_receiver.reg_z_pos, 'ro')
ax[1].plot(df_emitter.reg_x_pos, df_emitter.reg_z_pos, 'bo')
ax[1].set_ylim(150,0)
plt.savefig(os.path.join(Exp.savePath, Exp.runID, '{}_{}_emitter_receiver_cells.png'.format(Exp.fishID, Exp.runID)))

all_cells = df.cell_id.unique()
emitter_cells = [i for i,j in enumerate(all_cells) if j in df_emitter.cell_id.unique()]
np.save(os.path.join(Exp.savePath, Exp.runID, 'emitter_cells.npy'), emitter_cells)
receiver_cells = df_receiver.cell_id.unique()
np.save(os.path.join(Exp.savePath, Exp.runID, 'receiver_cells.npy'), receiver_cells)
print('n emitter: {}, n receiver: {}'.format(len(emitter_cells), len(receiver_cells)))
