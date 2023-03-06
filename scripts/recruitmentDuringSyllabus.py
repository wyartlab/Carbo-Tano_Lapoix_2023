import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import traceback
import shelve
import pyabf
from utils.calcium_traces import get_pos_y, get_pos_x
from utils.groupsCells import get_cells_group, get_cell_side, get_cells_group_added
from utils.ref_pos import get_ref_pos, addRefPos, addRefPos_rotate
from utils.get_ci_vs_bh import get_syl_isolation
from utils.behavior_vs_stimulation import get_bout_period


# For Carbo-Tano, Lapoix 2022
# Dataset: MLR electrical stim, 40s, 0.1uA, while recording vsx2+ calcium acitivty alongside tail angle movement
# Goal: quantify recruitment of vsx2+ cells during forward locomotion


def get_max_dff(syl, cell, df_syl, dff, fps_bh, fps_ci):
    start_syl, end_syl = df_syl.start[syl] / fps_bh, df_syl.end[syl] / fps_bh
    start = math.floor(start_syl * fps_ci)
    end = math.ceil(end_syl * fps_ci)
    max_dff = np.nanmax(dff[cell, start:end + 2])

    return max_dff


def get_index_max_dff(syl, cell, df_syl, dff, fps_bh, fps_ci):
    start_syl, end_syl = df_syl.start[syl] / fps_bh, df_syl.end[syl] / fps_bh
    start = math.floor(start_syl * fps_ci)
    end = math.ceil(end_syl * fps_ci)
    output = np.nanargmax(dff[cell, start:end + 2]) + 1

    return output


def get_norm_max_dff(syl, cell, df_syl, dff, fps_bh, fps_ci):
    """

    Compares max DF/F during a swim event to the updated baseline DF/F just before this swimming event.
    Makes sure that the max DF/F you find during a swimming event is not linked to previous activation of the cell
    linked to previous swimming events (or other).
    Normalised baseline DF/F is taken as the median DF/F between 3 and 1 frame before the approximate start of the
    swimming event.

    :param syl: int, syllabus number
    :param cell: int, cell number
    :param df_syl: dataframe, timing and category info of syllabus
    :param dff: numpy array of with DF/F for each cell, shape nCells x nFrames
    :param fps_bh: float/int, acquisition rate of behavior camera
    :param fps_ci: float, int, acquisition rate of calcium imaging camera
    :return: float, normalised max DF/F

    """

    start_syl, end_syl = df_syl.start[syl] / fps_bh, df_syl.end[syl] / fps_bh
    start = math.floor(start_syl * fps_ci)
    end = math.ceil(end_syl * fps_ci)
    max_dff = np.nanmax(dff[cell, start:end + 2])
    norm_baseline = np.nanmedian(dff[cell, start - 3: start - 1])

    output = max_dff - norm_baseline

    return output


def get_recruitment(syl, cell, df_syl, dff, noise, fps_bh, fps_ci):
    """
    Defines if a cell was recruited during a specific syllabus, by looking at the max DF/F during the time of the
    syllabus and comparing it to threshold of 5*noise of the cell + normalised baseline DF/F just before the
    behavior event.

    :param syl: int, syllabus number
    :param cell: int, cell number
    :param df_syl: dataframe, timing and category info of syllabus
    :param dff: numpy array of with DF/F for each cell, shape nCells x nFrames
    :param noise: numpy array/list with noise value fotr each cell, shape nCells
    :param fps_bh: float/int, acquisition rate of behavior camera
    :param fps_ci: float, int, acquisition rate of calcium imaging camera
    :return: binary value, whether cell was recruited during syl


    """
    start_syl, end_syl = df_syl.start[syl] / fps_bh, df_syl.end[syl] / fps_bh
    start = math.floor(start_syl * fps_ci)
    end = math.ceil(end_syl * fps_ci)
    max_dff = np.nanmax(dff[cell, start:end + 2])
    norm_baseline = np.nanmedian(dff[cell, start - 3: start - 1])

    if max_dff >= 5 * noise[cell] + norm_baseline:
        output = 1
    else:
        output = 0

    return output


def calc_cell_max_all_syl(cell, df_syl, dff, fps_bh, fps_ci):
    """Calculate, for one cell, the max dff reached by this cell around each sylulation.
    The signal in which the max dff will be looked for is defined by windows."""

    return pd.Series(range(nSyl)).apply(get_max_dff, args=(cell, df_syl, dff, fps_bh, fps_ci))


def calc_cell_norm_max_all_syl(cell, df_syl, dff, fps_bh, fps_ci):
    """Calculate, for one cell, the max dff reached by this cell around each sylulation.
    The signal in which the max dff will be looked for is defined by windows."""

    return pd.Series(range(nSyl)).apply(get_norm_max_dff, args=(cell, df_syl, dff, fps_bh, fps_ci))


def calc_cell_index_max_all_syl(cell, df_syl, dff, fps_bh, fps_ci):
    """Calculate, for one cell, the max dff reached by this cell around each sylulation.
    The signal in which the max dff will be looked for is defined by windows."""

    return pd.Series(range(nSyl)).apply(get_index_max_dff, args=(cell, df_syl, dff, fps_bh, fps_ci))


def calc_cell_recruitment_all_syl(cell, df_syl, dff, noise, fps_bh, fps_ci):
    """Calculate if a cell was recruited during each sylulation.
    The signal in which the max dff will be looked for is defined by windows.

    Return pandas Series with 0 or 1 as recruitment for each sylulation."""

    return pd.Series(range(nSyl)).apply(get_recruitment,
                                        args=(cell, df_syl, dff, noise, fps_bh, fps_ci))


def assign_final_cell_group(cell, df):
    x = list(df.loc[df.cell == cell, 'norm_x_pos'])[0]
    plane = list(df.loc[df.cell == cell, 'real_plane'])[0]
    group = list(df.loc[df.cell == cell, 'group'])[0]
    if x <= -45:
        output = 'prepontine'
    elif (-45 < x <= 75) & (plane <= 220):
        output = 'pontine_ventral'
    elif (-45 < x <= 75) & (plane > 220):
        output = 'pontine_dorsal'
    elif (75 < x <= 173) & (plane <= 220):
        output = 'retropontine_ventral'
    elif (75 < x <= 173) & (plane > 220):
        output = 'retropontine_dorsal'
    else:
        if group == 'spinal_cord':
            output = 'spinal_cord'
        elif group in ['bulbar_lateral_contra', 'bulbar_lateral_ipsi']:
            output = 'medullar_lateral'
        else:
            output = 'medulla'
    return output


def assign_final_cell_group_split_medulla(cell, df):
    x = list(df.loc[df.cell == cell, 'norm_x_pos'])[0]
    plane = list(df.loc[df.cell == cell, 'real_plane'])[0]
    group = list(df.loc[df.cell == cell, 'group'])[0]
    if x <= -45:
        output = 'prepontine'
    elif (-45 < x <= 75) & (plane <= 220):
        output = 'pontine_ventral'
    elif (-45 < x <= 75) & (plane > 220):
        output = 'pontine_dorsal'
    elif (75 < x <= 173) & (plane <= 220):
        output = 'retropontine_ventral'
    elif (75 < x <= 173) & (plane > 220):
        output = 'retropontine_dorsal'
    else:
        if group == 'spinal_cord':
            output = 'spinal_cord'
        elif group in ['bulbar_lateral_contra', 'bulbar_lateral_ipsi']:
            output = 'medullar_lateral'
        else:
            if 173 < x < 250:
                output = 'rostral_medulla'
            else:
                output = 'caudal_medulla'
    return output


def get_real_i_tbf(df_frame, syllabus_mask):
    even_bend = False
    iTBF = []
    j = 0
    for bend in df_frame[syllabus_mask & (df_frame.Bend_Amplitude.notna())].index:
        if j == 0:
            iTBF.append(np.nan)
            previous_bend = bend
            j += 1
            continue
        if not even_bend:
            iTBF.append(np.nan)
            even_bend = True
        else:
            time_bend1 = float(df_frame.Time_index[previous_bend])
            time_bend2 = float(df_frame.Time_index[bend])
            iTBF.append(round(1 / (time_bend2 - time_bend1), 3))
            even_bend = False
            previous_bend = bend

    return iTBF


# Initialize

save_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/V2a_recruitment_behavior/analysis_11/'
df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/data_summary_BH.csv')
plt.style.use('seaborn-poster')

# Build dataframe for all experiments

dict_all = {}
cell_id_num = 0

for exp in df_summary.index:

    if df_summary.use[exp] == 0:
        continue

    # Load dataset
    try:
        # Load corresponding behavior
        output_path = df_summary.output_path[exp]
        fps_bh = df_summary.frameRateBeh[exp]
        fps_ci = df_summary.frameRate[exp]
        tail_angle = np.load(output_path + '/dataset/tail_angle.npy')
        df_bout = pd.read_pickle(output_path + '/dataset/df_bout')
        df_syl = pd.read_pickle(output_path + '/dataset/df_syllabus_manual')
        df_frame = pd.read_pickle(output_path + '/dataset/df_frame')
        time_indices_bh = np.load(output_path + '/dataset/time_indices.npy')
        fish = df_summary.fishlabel[exp]
        plane = df_summary.plane[exp]
        real_plane = df_summary.real_plane[exp]
        data_path = df_summary.data_path[exp]
        ops = np.load(data_path + '/suite2p/plane0/ops.npy', allow_pickle=True).item()
        abf = pyabf.ABF(df_summary.stim_trace_path.iloc[exp])
        # Get time at which behavior camera started
        channel_camera = [i for i, a in enumerate(abf.adcNames) if a in ['IN 0', 'IN 10', 'Behavior']][0]
        abf.setSweep(sweepNumber=0, channel=channel_camera)
        shift = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]
        channel_stim = [i for i, a in enumerate(abf.adcNames) if a in ['Stim', 'Stim_OUT']][0]
        abf.setSweep(sweepNumber=0, channel=channel_stim)
        print('\n\n{}, {},\n{}'.format(fish, plane, output_path))
    except FileNotFoundError:
        traceback.print_exc()
        continue

    with shelve.open(output_path + '/shelve_calciumAnalysis.out') as f:
        cells = f['cells']
        dff = f['dff_c']
        dff_f = f['dff_f_avg']
        noise = f['noise']
        noise_f = f['noise_f_avg']
        stat = f['stat']

    #  Compute missing information from dataset
    noise_f2 = np.zeros(noise.shape)
    noise_f2[:] = np.nan
    for i, cell in enumerate(cells):
        noise_f2[cell] = noise_f[i]

    #  Define if S labelled syllabus have a forward component & if syllabus is isolated
    nSyl = len(df_syl)
    df_syl['forward_component'] = 0
    df_syl['isolated'] = False
    df_syl['bout_condition'] = np.nan
    df_syl['median_bend_amps'] = np.nan
    df_syl['median_iTBF'] = np.nan
    for i in df_syl.index:
        if df_syl.Cat[i] == 'F':
            df_syl.forward_component[i] = 1
        else:
            bends = np.array(df_syl.bend_amps[i])
            nBends = len(bends)
            nBends_bellow = len(np.where(np.abs(bends) <= 25)[0])
            if (df_syl.duration[i] >= 1.5) & (nBends_bellow >= 0.6 * nBends):
                df_syl.forward_component[i] = 1
        df_syl.isolated[i] = get_syl_isolation(i, df_syl, 300, 0.3)
        df_syl.bout_condition[i] = get_bout_period(i, df_syl, time_indices_bh, abf,
                                                   shift, nStim=1, stim_dur=40)

        syllabus_mask = df_frame.index.isin(np.arange(df_syl.start[i], df_syl.end[i] + 1))
        iTBF = get_real_i_tbf(df_frame, syllabus_mask)
        df_syl.iTBF[i] = iTBF
        df_syl.median_iTBF[i] = np.nanmedian(df_syl.iTBF[i])
        df_syl.mean_iTBF[i] = np.nanmean(df_syl.iTBF[i])

        df_syl.bend_amps[i] = list(df_frame.loc[syllabus_mask & (df_frame.Bend_Amplitude.notna()),
                                                                 'Bend_Amplitude'])
        df_syl.median_bend_amps[i] = np.nanmedian(df_syl.bend_amps[i])
        df_syl.median_iTBF[i] = np.nanmedian(df_syl.iTBF[i])


    #  Assign cells side and group
    sides = np.array(get_cell_side(df_summary, fish, plane, dff, cells, stat))
    a, _, _, _, _, _ = get_cells_group(df_summary, fish, plane, dff, cells, stat)
    groups = np.array(a)

    #   Build DataFrame
    df_lf = pd.DataFrame({'cell': np.repeat(cells, nSyl),
                          'syl': np.tile(list(range(nSyl)), len(cells)),
                          'syl_cat': np.tile(list(df_syl.Cat), len(cells)),
                          'syl_forward_component': np.tile(list(df_syl.forward_component), len(cells)),
                          'syl_max_bend': np.tile(list(df_syl.abs_max_bend_amp), len(cells)),
                          'syl_side': np.tile(list(df_syl.max_bend_amp), len(cells)),
                          'syl_duration': np.tile(list(df_syl.duration), len(cells)),
                          'syl_n_osc': np.tile(list(df_syl.n_oscillations), len(cells)),
                          'syl_isolated': np.tile(list(df_syl.isolated), len(cells)),
                          'syl_condition': np.tile(list(df_syl.bout_condition), len(cells)),
                          'syl_median_iTBF': np.tile(list(df_syl.median_iTBF), len(cells)),
                          'syl_mean_iTBF': np.tile(list(df_syl.mean_iTBF), len(cells)),
                          'syl_median_bend_amp': np.tile(list(df_syl.median_bend_amps), len(cells)),
                          'x_pos': np.repeat(pd.Series(cells).apply(get_pos_x, args=(stat,)), nSyl),
                          'y_pos': np.repeat(pd.Series(cells).apply(get_pos_y, args=(stat,)), nSyl),
                          'side': np.repeat(sides[cells], nSyl),
                          'group': np.repeat(groups[cells], nSyl),
                          'fishlabel': [fish] * len(np.repeat(cells, nSyl)),
                          'plane': [plane] * len(np.repeat(cells, nSyl)),
                          'real_plane': [real_plane] * len(np.repeat(cells, nSyl)),
                          'noise_f': np.repeat(noise_f, nSyl),
                          'max_dff': [0] * len(np.repeat(cells, nSyl)),
                          'max_dff_f': [0] * len(np.repeat(cells, nSyl)),
                          'norm_max_dff_f': [0] * len(np.repeat(cells, nSyl)),
                          'index_max_dff_f': [0] * len(np.repeat(cells, nSyl)),
                          'recruitment_f': [0] * len(np.repeat(cells, nSyl))

                          })

    # fill the max DF/F column
    maxs = pd.Series(cells).apply(calc_cell_max_all_syl,
                                  args=(df_syl, dff, fps_bh, fps_ci))
    df_lf['max_dff'] = np.array(maxs).flatten()

    # fill max DF/F filtered column
    maxs_filtered = pd.Series(cells).apply(calc_cell_max_all_syl,
                                           args=(df_syl, dff_f, fps_bh, fps_ci))
    df_lf['max_dff_f'] = np.array(maxs_filtered).flatten()

    # norm max DF/F
    norm_maxs_filtered = pd.Series(cells).apply(calc_cell_norm_max_all_syl,
                                                args=(df_syl, dff_f, fps_bh, fps_ci))
    df_lf['norm_max_dff_f'] = np.array(norm_maxs_filtered).flatten()


    #  index max DF/F
    indices_max_f = pd.Series(cells).apply(calc_cell_index_max_all_syl,
                                           args=(df_syl, dff_f, fps_bh, fps_ci))
    df_lf['index_max_dff_f'] = np.array(indices_max_f).flatten()

    #  recruitment
    recruitment_f = pd.Series(cells).apply(calc_cell_recruitment_all_syl,
                                           args=(df_syl, dff_f, noise_f2, fps_bh, fps_ci))
    df_lf['recruitment_f'] = np.array(recruitment_f).flatten()

    #  add ref pos
    x_ref, y_ref = df_summary.sc_bulbar[0], df_summary.midline[1]
    if df_summary.direction[exp] == 0:
        df_lf = addRefPos(fish, df_summary, x_ref, y_ref, df_lf)
    else:
        df_lf = addRefPos_rotate(fish, df_summary, x_ref, y_ref, df_lf)

    # Define recruitment and assign final cell group
    df_lf['final_cell_group'] = np.nan
    df_lf['final_cell_group_split_medulla'] = np.nan
    df_lf['cell_id'] = np.nan

    previous_positions = []
    for cell in cells:
        df_lf.loc[df_lf.cell == cell, 'mean_recruitment_F_syl'] = np.nanmean(
            df_lf.loc[(df_lf.cell == cell) & (df_lf.syl_cat == 'F'), 'recruitment_f'])
        df_lf.loc[df_lf.cell == cell, 'mean_recruitment_F_component'] = np.nanmean(
            df_lf.loc[(df_lf.cell == cell) & (df_lf.syl_forward_component == 1), 'recruitment_f'])

        df_lf.loc[df_lf.cell == cell, 'mean_recruitment_S_component'] = np.nanmean(
            df_lf.loc[(df_lf.cell == cell) & (df_lf.syl_forward_component == 0), 'recruitment_f'])
        df_lf.loc[df_lf.cell == cell, 'mean_recruitment_S_component_L'] = np.nanmean(
            df_lf.loc[(df_lf.cell == cell) & (df_lf.syl_forward_component == 0) & (df_lf.syl_side > 0),
                      'recruitment_f'])
        df_lf.loc[df_lf.cell == cell, 'mean_recruitment_S_component_R'] = np.nanmean(
            df_lf.loc[(df_lf.cell == cell) & (df_lf.syl_forward_component == 0) & (df_lf.syl_side < 0),
                      'recruitment_f'])

        df_lf.loc[df_lf.cell == cell, 'final_cell_group'] = assign_final_cell_group(cell, df_lf)
        df_lf.loc[df_lf.cell == cell, 'final_cell_group_split_medulla'] = assign_final_cell_group_split_medulla(cell,
                                                                                                                df_lf)

        df_lf.loc[df_lf.cell == cell, 'cell_id'] = cell_id_num

        previous_positions.append((stat[cell]['med'][0], stat[cell]['med'][1]))

        cell_id_num += 1

    #  Add manual cells

    new_stat = np.load(data_path + '/suite2p/plane0/stat.npy', allow_pickle=True)
    new_iscell = np.load(data_path + '/suite2p/plane0/iscell.npy', allow_pickle=True)
    new_cells = np.flatnonzero(new_iscell[:, 0])

    added_cells = []
    added_cells_id = []

    for cell in new_cells:
        if (new_stat[cell]['med'][0], new_stat[cell]['med'][1]) not in previous_positions:
            added_cells.append(cell)
            added_cells_id.append(cell_id_num)
            cell_id_num += 1

    sides = np.array(get_cell_side(df_summary, fish, plane, new_iscell, added_cells, new_stat))
    a, _, _, _, _, _ = get_cells_group_added(df_summary, fish, plane, new_iscell, added_cells, new_stat)
    groups_added = np.array(a)

    df_lf_added = pd.DataFrame({'cell': np.repeat([i + 1000 for i in added_cells], nSyl),
                                'syl': np.tile(list(range(nSyl)), len(added_cells)),
                                'syl_cat': np.tile(list(df_syl.Cat), len(added_cells)),
                                'syl_forward_component': np.tile(list(df_syl.forward_component), len(added_cells)),
                                'syl_max_bend': np.tile(list(df_syl.abs_max_bend_amp), len(added_cells)),
                                'syl_side': np.tile(list(df_syl.max_bend_amp), len(added_cells)),
                                'syl_duration': np.tile(list(df_syl.duration), len(added_cells)),
                                'syl_n_osc': np.tile(list(df_syl.n_oscillations), len(added_cells)),
                                'syl_isolated': np.tile(list(df_syl.isolated), len(added_cells)),
                                'syl_condition': np.tile(list(df_syl.bout_condition), len(added_cells)),
                                'x_pos': np.repeat(pd.Series(added_cells).apply(get_pos_x, args=(new_stat,)), nSyl),
                                'y_pos': np.repeat(pd.Series(added_cells).apply(get_pos_y, args=(new_stat,)), nSyl),
                                'side': np.nan,
                                'group': np.repeat(groups_added[added_cells], nSyl),
                                'fishlabel': [fish] * len(added_cells) * nSyl,
                                'plane': [plane] * len(added_cells) * nSyl,
                                'real_plane': [real_plane] * len(added_cells) * nSyl,
                                'noise_f': np.nan,
                                'max_dff': 0,
                                'max_dff_f': 0,
                                'norm_max_dff_f': 0,
                                'index_max_dff_f': 0,
                                'recruitment_f': 0,
                                'added': 1,
                                'cell_id': np.repeat(added_cells_id, nSyl)

                                })

    if df_summary.direction[exp] == 0:
        df_lf_added = addRefPos(fish, df_summary, x_ref, y_ref, df_lf_added)
    else:
        df_lf_added = addRefPos_rotate(fish, df_summary, x_ref, y_ref, df_lf_added)

    for cell in set(df_lf_added.cell):
        df_lf_added.loc[df_lf_added.cell == cell, 'final_cell_group'] = assign_final_cell_group(cell, df_lf_added)
        df_lf_added.loc[df_lf_added.cell == cell, 'final_cell_group_split_medulla'] = assign_final_cell_group_split_medulla(cell, df_lf_added)

    dict_all[exp] = pd.concat([df_lf, df_lf_added], ignore_index=True)

df = pd.concat(dict_all, ignore_index=True)

df.to_pickle(save_path + '/df_with_spinal_cord.pkl')
#  Remove spinal cord cells
df_all_cells = df.copy()
df = df[df.final_cell_group != 'spinal_cord']

# Plot mean recruitment during different syllabus categories

ops_ref = np.load(
    '/network/lustre/iss01/wyart/rawdata/2pehaviour/MLR/Calcium_Imaging/210121/F05/70um_bh/suite2p/plane0/ops.npy',
    allow_pickle=True).item()

#  Compute regression link between normalised max DF/F and bout duration/ number of oscillations

df['slope_dff_dur_F_comp'] = np.nan
df['pvalue_dff_dur_F_comp'] = np.nan

for cell in set(df.cell_id):

    try:

        # dff vs number of oscillations for pure forward
        x = df[(df.cell_id == cell) & (df.syl_cat == 'F')].syl_n_osc
        y = df[(df.cell_id == cell) & (df.syl_cat == 'F')].norm_max_dff_f

        slope, intercept, r, p, se = linregress(x, y)

        df.at[(df.cell_id == cell), 'slope_dff_osc_F'] = slope
        df.at[(df.cell_id == cell), 'pvalue_dff_osc_F'] = p

        # dff vs syllabus duration for pure forward
        x = df[(df.cell_id == cell) & (df.syl_cat == 'F')].syl_duration
        y = df[(df.cell_id == cell) & (
                df.syl_cat == 'F')].norm_max_dff_f

        slope, intercept, r, p, se = linregress(x, y)

        df.at[(df.cell_id == cell), 'slope_dff_dur_F'] = slope
        df.at[(df.cell_id == cell), 'pvalue_dff_dur_F'] = p

        # dff vs duration for forward component
        x = df[(df.cell_id == cell) & (df.syl_forward_component == 1)].syl_duration
        y = df[(df.cell_id == cell) & (df.syl_forward_component == 1)].norm_max_dff_f

        slope, intercept, r, p, se = linregress(x, y)

        df.at[(df.cell_id == cell), 'slope_dff_dur_F_comp'] = slope
        df.at[(df.cell_id == cell), 'pvalue_dff_dur_F_comp'] = p

    except ValueError:
        pass

# Select cells who have significant slope between how much they were recruited and how long the bout was
alpha = 0.05
mask_sig_slope_dur = df.pvalue_dff_dur_F <= alpha
#  Select cells whose slope is positive
mask_pos_slope_dur = df.slope_dff_dur_F > 0

#  Plot log of norm max DF/F and bout duration only for those cells
df['log_norm_max'] = np.log10(df['norm_max_dff_f'])
df['log_dur'] = np.log10(df['syl_duration'])
ax = sns.lmplot(data=df[mask_sig_slope_dur & mask_pos_slope_dur &
                        (df.final_cell_group.isin(['rostral_medulla', 'caudal_medulla',
                                                   'medullar_lateral', 'retropontine_dorsal']))],
                x='log_dur', y='log_norm_max', hue='final_cell_group', palette='Set1')

# Find forward specific cells

df['forward_specific'] = False
df['forward_specific_absolute'] = False
df['forward_component_specific'] = False
for cell in set(df.cell_id):
    n_forward = len(df[(df.cell_id == cell) & (df.syl_cat == 'F')])
    n_recruitment_f = len(df[(df.cell_id == cell) & (df.syl_cat == 'F') & (df.recruitment_f == 1)])
    if n_forward == 0:
        df.at[df.cell_id == cell, 'forward_specific'] = np.nan
    elif n_recruitment_f / n_forward >= 0.75:
        df.at[df.cell_id == cell, 'forward_specific'] = True
    elif n_recruitment_f == n_forward:
        df.at[df.cell_id == cell, 'forward_specific_absolute'] = True

    n_forward_comp = len(df[(df.cell_id == cell) & (df.syl_forward_component == 1)])
    n_recruitment_f_comp = len(df[(df.cell_id == cell) & (df.syl_forward_component == 1) & (df.recruitment_f == 1)])
    if n_forward_comp == 0:
        df.at[df.cell_id == cell, 'forward_component_specific'] = np.nan
    elif n_recruitment_f_comp / n_forward_comp == 1:
        df.at[df.cell_id == cell, 'forward_component_specific'] = True

print('N cells F specific:', len(set(df[df.forward_specific == True].cell_id)))
print('N cells F component specific:', len(set(df[df.forward_component_specific == True].cell_id)))

## Plot forward specific cells

# fig, ax = plt.subplots(1, 3, figsize=(12, 12))
# fig.suptitle('Forward specific cells')
# ax[0].imshow(ops_ref['meanImg'], cmap='Greys')
# ax[0].set_title('Cells always recruited for forward bouts')
# sns.scatterplot(data=df[(df.forward_specific == True) & (df.syl == 0)], x='norm_y_pos', y='norm_x_pos',
#                 style='fishlabel', palette='plasma', hue='final_cell_group', ax=ax[0])
#
# ax[0].get_legend().remove()
# ax[1].imshow(ops_ref['meanImg'], cmap='Greys')
# ax[1].set_title('Cells always recruited during bouts with forward components')
# sns.scatterplot(data=df[(df.forward_component_specific == True) & (df.syl == 0)], x='norm_y_pos', y='norm_x_pos',
#                 style='fishlabel', palette='plasma', hue='final_cell_group', ax=ax[1])
# ax[1].get_legend().remove()
# ax[2].imshow(ops_ref['meanImg'], cmap='Greys')
# sns.scatterplot(data=df[(df.forward_component_specific == True) & (df.forward_specific == True) & (df.syl == 0)],
#                 x='norm_y_pos', y='norm_x_pos',
#                 style='fishlabel', palette='plasma', hue='final_cell_group', ax=ax[2])
# ax[2].legend(bbox_to_anchor=(1.05, 1))
# ax[2].set_title('Intersection')

# Check you have F specific cells from different planes
print(set(df[mask_sig_slope_dur & mask_pos_slope_dur & (df.forward_specific == True)].fishlabel))

#  Plot F specific cells, with positive signficiant slope

# fig, ax = plt.subplots()
# ax.imshow(ops_ref['meanImg'], cmap='Greys')
# sns.scatterplot(data=df[mask_sig_slope_dur & mask_pos_slope_dur &
#                         (df.forward_specific is True) & (df.syl == 0)],
#                 x='norm_y_pos', y='norm_x_pos', palette='plasma', ax=ax)


# fig, ax = plt.subplots()
# ax.imshow(ops_ref['meanImg'], cmap='Greys')
# sns.scatterplot(data=df[(df.pvalue_dff_osc_F <= 0.5) & (df.forward_specific == True)], x='norm_y_pos', y='norm_x_pos',
#                 hue='slope_dff_osc_F',
#                 palette='plasma')
# ax.legend(bbox_to_anchor=(1.05, 1))
# fig.suptitle('Significant slope between max DF/F during F syllabus and n oscillations\nn=2 fish, only F specific '
#              'cells')
# plt.savefig('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/V2a_recruitment_behavior/analysis_6'
#             '/slope_dff_osc.svg')


# for fish in set(df.fishlabel):
#     fig, ax = plt.subplots(1, 2, figsize=(12, 12))
#     fig.suptitle(fish)
#     bg = np.load(list(df_summary[df_summary.fishlabel == fish].data_path)[0] + '/suite2p/plane0/ops.npy',
#                  allow_pickle=True).item()['meanImg']
#     ax[0].imshow(bg, cmap='Greys')
#     ax[1].imshow(bg, cmap='Greys')
#     ax[0].set_title('Mean recruitment F syllabus')
#     ax[1].set_title('Mean recruitment F component')
#     sns.scatterplot(data=df[df.fishlabel == fish], x='y_pos', y='x_pos', hue='mean_recruitment_F_syl', ax=ax[0])
#     sns.scatterplot(data=df[df.fishlabel == fish], x='y_pos', y='x_pos', hue='mean_recruitment_F_component', ax=ax[1])
#     ax[0].legend(bbox_to_anchor=(1.05, 1))
#     ax[1].legend(bbox_to_anchor=(1.05, 1))
#     plt.tight_layout()

# TODO: cherry pick experiments where we can compute a nice slope for medial-medullary neurons

cherry_fish = ['210121_F05', '210121_F04']
df_cherry = df[df.fishlabel.isin(cherry_fish)].copy()
for cell in set(df_cherry.cell_id):
    if len(df_cherry[(df_cherry.cell_id == cell) & (df_cherry.syl_cat == 'F')]) == 0:
        continue
    n_recruitment_f = len(df_cherry[(df_cherry.cell_id == cell) & (df_cherry.syl_cat == 'F') & (df_cherry.recruitment_f == 1)])
    if n_recruitment_f == len(df_cherry[(df_cherry.cell_id == cell) & (df_cherry.syl_cat == 'F')]):
        df_cherry.at[df_cherry.cell_id == cell, 'forward_specific_absolute'] = True

df_cherry['log_dur'] = np.log10(df_cherry.syl_duration)
sns.lmplot(data=df_cherry[df_cherry.forward_specific_absolute == True],
           x='log_dur', y='norm_max_dff_f', hue='final_cell_group')
plt.savefig(save_path + 'fig_dff_duration.svg')

for cell in set(df_cherry.cell_id):
    # dff vs number of oscillations for pure forward
    x = np.log(df_cherry[(df_cherry.cell_id == cell) & (df_cherry.syl_cat == 'F')].syl_n_osc)
    y = df_cherry[(df_cherry.cell_id == cell) & (df_cherry.syl_cat == 'F')].norm_max_dff_f

    slope, intercept, r, p, se = linregress(x, y)

    df_cherry.at[(df_cherry.cell_id == cell), 'slope_dff_log_osc_F'] = slope
    df_cherry.at[(df_cherry.cell_id == cell), 'pvalue_dff_log_osc_F'] = p


#  Proportion of cells recruited, in each groups, for each syllabus type


def prop_cells_recruited(syl, group, fish, plane, df):
    """
    Computes the proportion of cells in one anatomical group recruited during a specific syllabus.
    If, and only if, this syllabus is either a pure forward, or a struggle with no forward component.

    :param syl: syllabus number
    :param group: anatomical group of cells
    :param fish: fishlabel
    :param plane: plane
    :param df: dataframe with recruitment information, all experiments concatenated together
    :return: proportion of cells, out of all cells in a given anatomical group, recruited during the syllabus
    """

    exp_mask = (df.fishlabel == fish) & (df.plane == plane)
    n_Cells = len(df[exp_mask & (df.syl == 0) & (df.final_cell_group == group)])

    if (set(df[exp_mask & (df.syl == syl)].syl_cat) == {'F'}) or \
            (set(df[exp_mask & (df.syl == syl)].syl_forward_component) == {0}):
        output = len(df[exp_mask & (df.final_cell_group == group) & (df.syl == syl) & (
                df.recruitment_f == 1)]) / n_Cells

    else:
        output = np.nan

    return output


def prop_cells_recruited2(syl, group, fish, plane, df):
    """
    Computes the proportion of cells in one anatomical group recruited during a specific syllabus.
    If, and only if, this syllabus is either a pure forward, or a struggle with no forward component.

    :param syl: syllabus number
    :param group: anatomical group of cells
    :param fish: fishlabel
    :param plane: plane
    :param df: dataframe with recruitment information, all experiments concatenated together
    :return: proportion of cells, out of all cells in a given anatomical group, recruited during the syllabus
    """

    exp_mask = (df.fishlabel == fish) & (df.plane == plane)
    n_Cells = len(df[exp_mask & (df.syl == 0) & (df.final_cell_group_split_medulla == group)])

    if (set(df[exp_mask & (df.syl == syl)].syl_cat) == {'F'}) or \
            (set(df[exp_mask & (df.syl == syl)].syl_forward_component) == {0}):
        output = len(df[exp_mask & (df.final_cell_group_split_medulla == group) & (df.syl == syl) & (
                df.recruitment_f == 1)]) / n_Cells

    else:
        output = np.nan

    return output


dict_prop = {}
syl_categories = list(set(df.syl_cat))
groups = list(set(df.final_cell_group).union(set(df.final_cell_group_split_medulla)))
n_syl_cat, n_groups = len(syl_categories), len(groups)

for exp in df_summary[df_summary.use == 1].index:

    fish, plane = df_summary.fishlabel[exp], df_summary.plane[exp]
    df_exp = pd.DataFrame({'fishlabel': fish,
                           'plane': plane,
                           'exp': exp,
                           'syl_cat': np.repeat(syl_categories, n_groups),
                           'group': np.tile(groups, n_syl_cat),
                           'prop': [np.nan],
                           'mean_prop_cells_recruited': np.nan,
                           'median_prop_cells_recruited': np.nan},

                          index=range(n_syl_cat * n_groups),
                          dtype=None)

    for group in groups:

        exp_mask = (df.fishlabel == fish) & (df.plane == plane)
        if group in ['rostral_medulla', 'caudal_medulla']:
            n_Cells = len(df[exp_mask & (df.syl == 0) & (df.final_cell_group_split_medulla == group)])
            if n_Cells == 0:
                continue

            for syl_cat in syl_categories:
                syls = list(set(df[exp_mask & (df.syl_cat == syl_cat) & (df.syl_condition == 'during_stim')].syl))
                prop = [prop_cells_recruited2(syl, group, fish, plane, df) for syl in syls]

                # df_exp.loc[(df_exp.group == group) & (df_exp.syl_cat == syl_cat),
                #           'prop'] = prop
                df_exp.loc[(df_exp.group == group) & (df_exp.syl_cat == syl_cat),
                           'mean_prop_cells_recruited'] = np.nanmean(prop)
                df_exp.loc[(df_exp.group == group) & (df_exp.syl_cat == syl_cat),
                           'median_prop_cells_recruited'] = np.nanmedian(prop)
        else:
            n_Cells = len(df[exp_mask & (df.syl == 0) & (df.final_cell_group == group)])
            if n_Cells == 0:
                continue

            for syl_cat in syl_categories:
                syls = list(set(df[exp_mask & (df.syl_cat == syl_cat) & (df.syl_condition == 'during_stim')].syl))
                prop = [prop_cells_recruited(syl, group, fish, plane, df) for syl in syls]

                # df_exp.loc[(df_exp.group == group) & (df_exp.syl_cat == syl_cat),
                #           'prop'] = prop
                df_exp.loc[(df_exp.group == group) & (df_exp.syl_cat == syl_cat),
                           'mean_prop_cells_recruited'] = np.nanmean(prop)
                df_exp.loc[(df_exp.group == group) & (df_exp.syl_cat == syl_cat),
                           'median_prop_cells_recruited'] = np.nanmedian(prop)

    dict_prop[exp] = df_exp

df_prop = pd.concat(dict_prop, ignore_index=True)

#  Plot distribution of resulting proportions

fig, ax = plt.subplots(figsize=(8, 12))
sns.boxplot(data=df_prop[df_prop.group != 'pontine_dorsal'], x='group', y='mean_prop_cells_recruited', hue='syl_cat',
            order=['medullar_lateral', 'medulla', 'retropontine_dorsal'],
            hue_order=['F', 'S'],
            linewidth=2.5,
            ax=ax, palette='muted')
sns.swarmplot(data=df_prop[df_prop.group != 'pontine_dorsal'], x='group', y='mean_prop_cells_recruited', hue='syl_cat',
              split=True,
              linewidth=1, edgecolor='gray', hue_order=['F', 'S'],
              order=['medullar_lateral', 'medulla', 'retropontine_dorsal'],

              ax=ax, palette='muted')

#  Computing proportions by grouping the different planes from one fish together
dict_prop_fish = {}
for fish in set(df.fishlabel):

    df_fish = pd.DataFrame({'fishlabel': fish,
                            'syl_cat': np.repeat(syl_categories, n_groups),
                            'group': np.tile(groups, n_syl_cat),
                            'prop': [np.nan],
                            'mean_prop_cells_recruited': np.nan,
                            'median_prop_cells_recruited': np.nan},

                           index=range(n_syl_cat * n_groups),
                           dtype=None)

    for group in groups:

        if group in ['rostral_medulla', 'caudal_medulla']:
            n_Cells = len(df[exp_mask & (df.syl == 0) & (df.final_cell_group_split_medulla == group)])
            if n_Cells == 0:
                continue

            for syl_cat in syl_categories:

                prop_fish = []

                for exp in df_summary[(df_summary.use == 1) & (df_summary.fishlabel == fish)].index:

                    plane = df_summary.plane[exp]
                    exp_mask = (df.fishlabel == fish) & (df.plane == plane)

                    if not len(df[exp_mask & (df.syl == 0) & (df.final_cell_group_split_medulla == group)]) == 0:
                        syls = list(
                            set(df[exp_mask & (df.syl_cat == syl_cat) & (df.syl_condition == 'during_stim')].syl))
                        prop = [prop_cells_recruited2(syl, group, fish, plane, df) for syl in syls]
                        prop_fish = prop_fish + prop

                df_fish.loc[(df_fish.group == group) & (df_fish.syl_cat == syl_cat),
                            'mean_prop_cells_recruited'] = np.nanmean(prop_fish)
                df_fish.loc[(df_fish.group == group) & (df_fish.syl_cat == syl_cat),
                            'median_prop_cells_recruited'] = np.nanmedian(prop_fish)

        else:

            n_Cells = len(df[(df.fishlabel == fish) & (df.syl == 0) & (df.final_cell_group == group)])
            if n_Cells == 0:
                continue

            for syl_cat in syl_categories:

                prop_fish = []

                for exp in df_summary[(df_summary.use == 1) & (df_summary.fishlabel == fish)].index:

                    plane = df_summary.plane[exp]
                    exp_mask = (df.fishlabel == fish) & (df.plane == plane)

                    if not len(df[exp_mask & (df.syl == 0) & (df.final_cell_group == group)]) == 0:
                        syls = list(set(df[exp_mask & (df.syl_cat == syl_cat) & (df.syl_condition == 'during_stim')].syl))
                        prop = [prop_cells_recruited(syl, group, fish, plane, df) for syl in syls]
                        prop_fish = prop_fish + prop

                df_fish.loc[(df_fish.group == group) & (df_fish.syl_cat == syl_cat),
                            'mean_prop_cells_recruited'] = np.nanmean(prop_fish)
                df_fish.loc[(df_fish.group == group) & (df_fish.syl_cat == syl_cat),
                            'median_prop_cells_recruited'] = np.nanmedian(prop_fish)

        dict_prop_fish[fish] = df_fish

df_prop_fish = pd.concat(dict_prop_fish, ignore_index=True)

#  Plot distribution of resulting proportions
fig, ax = plt.subplots(figsize=(8, 12))
sns.boxplot(data=df_prop_fish[df_prop_fish.group != 'pontine_dorsal'], x='group', y='mean_prop_cells_recruited',
            hue='syl_cat',
            order=['medullar_lateral', 'medulla', 'retropontine_dorsal'],
            hue_order=['F', 'S'],
            linewidth=2.5,
            ax=ax, palette='muted')
sns.swarmplot(data=df_prop_fish[df_prop_fish.group != 'pontine_dorsal'], x='group', y='mean_prop_cells_recruited',
              hue='syl_cat', split=True,
              linewidth=1, edgecolor='gray', hue_order=['F', 'S'],
              order=['medullar_lateral', 'medulla', 'retropontine_dorsal'],

              ax=ax, palette='muted')
fig.autofmt_xdate(rotation=45)
plt.show()

# Only on isolated bout
#
# dict_prop_fish = {}
# for fish in set(df.fishlabel):
#
#     df_fish = pd.DataFrame({'fishlabel': fish,
#                             'syl_cat': np.repeat(syl_categories, n_groups),
#                             'group': np.tile(groups, n_syl_cat),
#                             'prop': [np.nan],
#                             'mean_prop_cells_recruited': np.nan,
#                             'median_prop_cells_recruited': np.nan},
#
#                            index=range(n_syl_cat * n_groups),
#                            dtype=None)
#
#     for group in groups:
#
#         n_Cells = len(df[(df.fishlabel == fish) & (df.syl == 0) & (df.final_cell_group == group)])
#         if n_Cells == 0:
#             continue
#
#         for syl_cat in syl_categories:
#
#             prop_fish = []
#
#             for exp in df_summary[(df_summary.use == 1) & (df_summary.fishlabel == fish)].index:
#
#                 plane = df_summary.plane[exp]
#                 exp_mask = (df.fishlabel == fish) & (df.plane == plane)
#
#                 if not len(df[exp_mask & (df.syl == 0) & (df.final_cell_group == group)]) == 0:
#                     syls = list(set(df[exp_mask & (df.syl_cat == syl_cat) & (df.syl_isolated == 1)].syl))
#                     prop = [prop_cells_recruited(syl, group, fish, plane, df) for syl in syls]
#                     prop_fish = prop_fish + prop
#
#             # df_exp.loc[(df_exp.group == group) & (df_exp.syl_cat == syl_cat),
#             #           'prop'] = prop
#             df_fish.loc[(df_fish.group == group) & (df_fish.syl_cat == syl_cat),
#                         'mean_prop_cells_recruited'] = np.nanmean(prop_fish)
#             df_fish.loc[(df_fish.group == group) & (df_fish.syl_cat == syl_cat),
#                         'median_prop_cells_recruited'] = np.nanmedian(prop_fish)
#
#         dict_prop_fish[fish] = df_fish
#
# df_prop_fish = pd.concat(dict_prop_fish, ignore_index=True)

# Save

df_prop.to_csv(save_path + 'df_prop.csv')
df_prop.to_pickle(save_path + 'df_prop.pkl')
df_prop_fish.to_csv(save_path + 'df_prop_fish.csv')
df_prop_fish.to_pickle(save_path + 'df_prop_fish.pkl')

#   Compute, for each cell, ratio between mean max activation during S or F

df['ratio_max_activation'] = np.nan
for cell in set(df.cell_id):
    if set(df[df.cell_id == cell].added) != {1}:
        mean_max_F = np.nanmean(df[(df.cell_id == cell) & (df.syl_cat == 'F') & (df.syl_condition == 'during_stim')].norm_max_dff_f)
        mean_max_S = np.nanmean(df[(df.cell_id == cell) & (df.syl_forward_component == 0) & (df.syl_condition == 'during_stim')].norm_max_dff_f)
        df.loc[df.cell_id == cell, 'ratio_max_activation'] = (mean_max_F - mean_max_S) / (mean_max_F + mean_max_S)

# plot ratio
plt.figure()
ax = sns.boxplot(data=df[df.syl == 0], y='ratio_max_activation', x='final_cell_group',
                 linewidth=2.5, showfliers=False,
                 palette='Pastel1',
                 order=['medullar_lateral', 'medulla', 'retropontine_dorsal'])
y_lim = ax.get_ylim()
sns.stripplot(data=df[df.syl == 0], y='ratio_max_activation', x='final_cell_group', jitter=True,
              linewidth=1, edgecolor='gray', palette='Pastel1', alpha=0.5,
              order=['medullar_lateral', 'medulla', 'retropontine_dorsal'])
ax.set_ylim(y_lim)

#  Final saving
df.to_pickle(save_path + 'df.pkl')
df.to_csv(save_path + 'df.csv')

# Plot example tail angle and calcium traces during spontaneous vs MLR - triggered forward bouts


from math import floor


exp = 12

try:
    # Load corresponding behavior
    output_path = df_summary.output_path[exp]
    fps_bh = df_summary.frameRateBeh[exp]
    fps_ci = df_summary.frameRate[exp]
    tail_angle = np.load(output_path + '/dataset/tail_angle.npy')
    df_frame = pd.read_pickle(output_path + '/dataset/df_frame')
    df_syl = pd.read_pickle(output_path + '/dataset/df_syllabus_manual')
    time_indices_bh = np.load(output_path + '/dataset/time_indices.npy')
    fish = df_summary.fishlabel[exp]
    plane = df_summary.plane[exp]
    real_plane = df_summary.real_plane[exp]
    data_path = df_summary.data_path[exp]
    ops = np.load(data_path + '/suite2p/plane0/ops.npy', allow_pickle=True).item()
    abf = pyabf.ABF(df_summary.stim_trace_path.iloc[exp])
    # Get time at which behavior camera started
    channel_camera = [i for i, a in enumerate(abf.adcNames) if a in ['IN 0', 'IN 10', 'Behavior']][0]
    abf.setSweep(sweepNumber=0, channel=channel_camera)
    shift = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]
    channel_stim = [i for i, a in enumerate(abf.adcNames) if a in ['Stim', 'Stim_OUT']][0]
    abf.setSweep(sweepNumber=0, channel=channel_stim)
    print('\n\n{}, {},\n{}'.format(fish, plane, output_path))
except FileNotFoundError:
    traceback.print_exc()

with shelve.open(output_path + '/shelve_calciumAnalysis.out') as f:
    cells = f['cells']
    dff = f['dff_c']
    dff_f = f['dff_f_avg']
    dff_f_lp = f['dff_f_lp']
    noise = f['noise']
    noise_f = f['noise_f_avg']
    stat = f['stat']

exp_mask = (df.fishlabel == fish) & (df.plane == plane)
time_indices_ci = np.arange(dff.shape[1])/fps_ci
syls = [0, 1, 2, 3]

# Find cells which respond at least once during those bouts
dict_cells = {}
for syl in syls:
    dict_cells[syl] = set(df[exp_mask & (df.syl == syl) & (df.recruitment_f == 1) &
                             (df.final_cell_group == 'medulla')].cell)


cells_to_plot = dict_cells[0].union(dict_cells[1], dict_cells[2], dict_cells[3])

# Plot
fig, ax = plt.subplots(figsize=(7, 12))
fig.suptitle(fish + '_' + plane)
time_inf, time_sup = 0, (6340/fps_bh)+2

bh_inf, bh_sup = int(time_inf*fps_bh), int(time_sup*fps_bh)
ax.plot(time_indices_bh[bh_inf:bh_sup], tail_angle[bh_inf:bh_sup]-20, color='black', linewidth=0.5)

ci_inf, ci_sup = floor(time_inf*fps_ci), floor(time_sup*fps_ci)
for i, cell in enumerate(cells_to_plot):
    try:
        ax.plot(time_indices_ci[ci_inf:ci_sup], dff_f_lp[int(cell),ci_inf:ci_sup]+i*10, linewidth=1)
    except IndexError:
        continue

ax.set_xlabel('Time [s]')
ax.set_ylabel('Tail angle [°] / DF/F [%]')
plt.tight_layout()
plt.savefig(save_path + 'example_spont.svg')

# plot maps of recruitment and DF/F during those bouts
fig_r, ax_r = plt.subplots(4,1,figsize=(7, 12))
fig_r.suptitle(fish + '_' + plane+'\nrecruitment')

fig_dff, ax_dff = plt.subplots(4,1,figsize=(7, 12))
fig_dff.suptitle(fish + '_' + plane+'\nmax DF/F')
for syl in syls:
    ax_r[syl].imshow(ops['meanImg'], cmap='Greys')
    sns.scatterplot(data=df[exp_mask & (df.syl == syl) & (df.cell.isin(cells_to_plot))],
                    x='y_pos', y='x_pos', hue='recruitment_f', ax=ax_r[syl])
    ax_dff[syl].imshow(ops['meanImg'], cmap='Greys')
    sns.scatterplot(data=df[exp_mask & (df.syl == syl) & (df.cell.isin(cells_to_plot))],
                    x='y_pos', y='x_pos', hue='norm_max_dff_f', hue_norm=(0,10),
                    ax=ax_dff[syl], palette='rocket_r')
fig_dff.savefig(save_path + '/dff_map.svg')
fig_r.savefig(save_path + '/recruitment_map.svg')


## How do these cells encode amplitude or tail beat frequency ?

from scipy.interpolate import interp1d

pos_bends_idx = np.array(df_frame[df_frame.Bend_Amplitude > 0].index)
pos_bends_amp = np.array(df_frame[df_frame.Bend_Amplitude > 0].Bend_Amplitude)
up_enveloppe = interp1d(pos_bends_idx, pos_bends_amp, kind='cubic', bounds_error=False, fill_value=0.0)
enveloppe = up_enveloppe(np.arange(len(tail_angle)))
enveloppe[np.where(tail_angle == 0)] = 0

fig, ax = plt.subplots()
ax.plot(time_indices_bh, tail_angle, color='silver', linewidth=0.5)
ax.plot(time_indices_bh, enveloppe, color='coral', linewidth=0.7)
ax.plot(time_indices_ci, dff_f_lp[60,:]+100, color='royalblue', linewidth=0.7)

# TODO: look at bins of time and compute max ta in each bin
# Compute derivative of max ta
bin_angle = np.zeros(time_indices_ci.shape)
bin_freq = np.zeros(time_indices_ci.shape)

for frame, time in enumerate(time_indices_ci):
    if frame == len(time_indices_ci)-1:
        continue
    next_time = time_indices_ci[frame+1]
    bh_inf, bh_sup = int(time*fps_bh), int(next_time*fps_bh)
    bin_angle[frame] = np.abs(df_frame.Tail_angle[bh_inf:bh_sup]).max()
    n_osc = len(df_frame[(df_frame.index.isin(range(bh_inf, bh_sup))) & (df_frame.Bend_Amplitude.notna())])/2
    if n_osc != 0:
        bin_freq[frame] = n_osc * fps_ci

bin_angle_dev = [bin_angle[i+1]-bin_angle[i] for i in range(len(bin_angle)-1)]

plt.figure()
plt.plot(time_indices_ci, bin_freq)
plt.title('bin freq')
plt.plot(time_indices_ci, dff_f_lp[60,:]+100, color='royalblue', linewidth=0.7)
plt.figure()
plt.plot(time_indices_ci, bin_angle)
plt.plot(time_indices_ci[-1], bin_angle_dev)
plt.title('Bin angle')
plt.plot(time_indices_ci, dff_f_lp[60,:]+100, color='royalblue', linewidth=0.7)

