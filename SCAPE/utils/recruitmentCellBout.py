import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums, zscore
from random import sample
import itertools
from tqdm import tqdm as _tqdm
import multiprocessing as _mp
import functools as _ft


def correct_frames_lim(frames, f):
    output = [np.nan] * 2

    if frames[0] < 0:
        output[0] = 0
    elif frames[0] > len(f):
        output[0] = np.nan
    else:
        output[0] = frames[0]

    if frames[1] < 0:
        output[1] = np.nan
    elif frames[1] > len(f):
        output[1] = -1
    else:
        output[1] = frames[1]

    return output


def get_frame_limits(bout, df_bout, Exp, Cell, lim_inf, lim_sup):
    time_inf = df_bout.BoutStart[bout] / Exp.frameRateBeh - lim_inf
    time_sup = df_bout.BoutEnd[bout] / Exp.frameRateBeh + lim_sup
    frame_inf, frame_sup = math.floor(time_inf * Exp.frameRateSCAPE), math.ceil(time_sup * Exp.frameRateSCAPE)
    frames = correct_frames_lim([frame_inf, frame_sup], Cell.dff)

    return frames


def get_norm_baseline(Exp, df_bout, bout, Cell, lim_inf=1, lim_sup=0.2):
    time_inf = df_bout.BoutStart[bout] / Exp.frameRateBeh - lim_inf
    time_sup = df_bout.BoutStart[bout] / Exp.frameRateBeh - lim_sup
    frame_inf, frame_sup = math.floor(time_inf * Exp.frameRateSCAPE), math.ceil(time_sup * Exp.frameRateSCAPE)
    frames = correct_frames_lim([frame_inf, frame_sup], Cell.dff)
    if any(np.isnan(frames)):
        output = np.nan
    else:
        output = np.nanmedian(Cell.dff_corrected[frames[0]:frames[1]])

    return output


def get_max_DFF(Exp, df_bout, bout, Cell, lim_inf=0.2, lim_sup=0.2):
    frames = get_frame_limits(bout, df_bout, Exp, Cell, lim_inf, lim_sup)
    if any(np.isnan(frames)):
        output = np.nan
    else:
        output = np.nanmax(Cell.dff_corrected[frames[0]:frames[1]])

    return output


def get_norm_max_DFF(Exp, df_bout, bout, Cell, lim_inf=0.2, lim_sup=0.2):
    max_DFF = get_max_DFF(Exp, df_bout, bout, Cell, lim_inf, lim_sup)
    norm_baseline = get_norm_baseline(Exp, df_bout, bout, Cell)
    if any(np.isnan(i) for i in [max_DFF, norm_baseline]):
        output = np.nan
    else:
        output = max_DFF - norm_baseline

    return output


def get_recruitment_from_dff(Exp, df_bout, bout, Cell, threshold=5, lim_inf=0.2, lim_sup=0.2):
    norm_max_dff = get_norm_max_DFF(Exp, df_bout, bout, Cell, lim_inf, lim_sup)
    norm_baseline = get_norm_baseline(Exp, df_bout, bout, Cell)
    noise = Cell.noise

    if norm_max_dff > threshold * noise + norm_baseline:
        return 1

    else:
        return 0


def get_all_spks_during_bout(Exp, df_bout, bout, Cell, lim_inf=0.2, lim_sup=0):
    frame_inf, frame_sup = get_frame_limits(bout, df_bout, Exp, Cell, lim_inf, lim_sup)
    if frame_inf < 0:
        frame_inf = 0
    if frame_sup > len(Cell.spks):
        frame_sup = len(Cell.spks)
    return Cell.spks[frame_inf:frame_sup]


def get_max_spks(Exp, df_bout, bout, Cell, lim_inf=0.2, lim_sup=0):
    spks = get_all_spks_during_bout(Exp, df_bout, bout, Cell, lim_inf, lim_sup)
    return np.nanmax(spks)


def get_mean_spks(Exp, df_bout, bout, Cell, lim_inf=0.2, lim_sup=0):
    spks = get_all_spks_during_bout(Exp, df_bout, bout, Cell, lim_inf, lim_sup)
    return np.nanmean(spks)


def get_recruit_from_spks(Exp, df_bout, bout, Cell, lim_inf=0.2, lim_sup=0):
    max_spks = get_max_spks(Exp, df_bout, bout, Cell, lim_inf, lim_sup)
    baseline = 2 * np.nanmean(get_spks_during_no_mov(Cell, Exp.ta_resampled))

    if max_spks >= baseline:
        return 1

    else:
        return 0


def get_mean_spks_during_bouts(Exp, df_bout, Cell):
    return [get_mean_spks(Exp, df_bout, bout, Cell) for bout in df_bout.index]


def get_all_spks_during_all_bouts(Exp, df_bout, Cell):
    for i, bout in enumerate(df_bout.index):
        if i == 0:
            output = np.array(get_all_spks_during_bout(Exp, df_bout, bout, Cell))
        else:
            output = np.hstack((output, get_all_spks_during_bout(Exp, df_bout, bout, Cell)))

    return output


def get_mean_spks_during_forward(Exp, df_bouts, Cell):
    return get_mean_spks_during_bouts(Exp, df_bouts[df_bouts.abs_Max_Bend_Amp <= 25], Cell)


def get_all_spks_during_forward(Exp, df_bouts, Cell):
    return get_all_spks_during_all_bouts(Exp, df_bouts[df_bouts.abs_Max_Bend_Amp <= 25], Cell)


def get_spks_during_no_mov(Cell, tail_angle_resampled):
    indices_no_mov = tail_angle_resampled == 0
    output = Cell.spks.copy()[indices_no_mov]
    return output


def build_df_recruitment(Cells, df_bout, Exp, limits_dff=(0.2, 0.2), limits_spks=(0.2, 0)):
    nBouts, nCells = len(df_bout), len(Cells)

    df_recruitment = pd.DataFrame({'cell_id': np.repeat([i.cellID for i in Cells], nBouts),
                                   'cell_num': np.repeat([i.init_cellID for i in Cells], nBouts),
                                   'bout_id': np.tile(df_bout.index, nCells),
                                   'bout_num': np.tile(df_bout.NumBout, nCells),
                                   'x_pos': np.repeat([i.x_pos for i in Cells], nBouts),
                                   'y_pos': np.repeat([i.y_pos for i in Cells], nBouts),
                                   'norm_y_pos': np.repeat([i.norm_y_pos for i in Cells], nBouts),
                                   'side': np.repeat([i.side for i in Cells], nBouts),
                                   'plane': np.repeat([i.plane for i in Cells], nBouts),
                                   'group': np.repeat([i.group for i in Cells], nBouts)})

    for key in ['spks', 'spks_no_mov', 'spks_during_forward']:
        df_recruitment[key] = pd.Series([np.nan] * len(df_recruitment)).astype('object')

    for key in ['Bout_Duration', 'Number_Osc', 'abs_Max_Bend_Amp', 'Max_Bend_Amp', 'mean_TBF',
                'median_iTBF', 'max_iTBF', 'median_bend_amp', 'mean_tail_angle']:
        df_recruitment[key] = np.tile(df_bout[key], nCells)

    ## compute recruitment for all cells, all bouts

    for i, Cell in enumerate(Cells):
        cell_mask = df_recruitment.cell_id == Cell.cellID

        for bout in df_bout.index:
            bout_mask = df_recruitment.bout_id == bout
            index = df_recruitment[cell_mask & bout_mask].index[0]

            df_recruitment.loc[cell_mask & bout_mask, 'norm_max_dff'] = get_norm_max_DFF(Exp, df_bout,
                                                                                         bout, Cell,
                                                                                         lim_inf=limits_dff[0],
                                                                                         lim_sup=limits_dff[1])
            df_recruitment.loc[cell_mask & bout_mask, 'recruitment'] = get_recruitment_from_dff(Exp, df_bout, bout,
                                                                                                Cell,
                                                                                                lim_inf=limits_dff[0],
                                                                                                lim_sup=limits_dff[1])
            # df_recruitment.loc[cell_mask & bout_mask, 'recruitment_deconv'] = get_recruit_from_spks(Exp, df_bout,
            #                                                                                         bout, Cell)

            col = np.where(df_recruitment.columns == 'spks')[0][0]
            df_recruitment.iat[index, col] = get_all_spks_during_bout(Exp, df_bout, bout, Cell,
                                                                      lim_inf=limits_spks[0],
                                                                      lim_sup=limits_spks[1])
            df_recruitment.loc[cell_mask & bout_mask, 'max_spks'] = [np.nanmax(i) for i in
                                                                     df_recruitment.loc[cell_mask & bout_mask, 'spks']]
            df_recruitment.loc[cell_mask & bout_mask, 'mean_spks'] = [np.nanmean(i) for i in
                                                                      df_recruitment.loc[cell_mask & bout_mask, 'spks']]
            # df_recruitment.loc[cell_mask & bout_mask, 'max_spks'] = get_max_spks(Exp, df_bout, bout, Cell,
            #                                                                     lim_inf=lim_inf, lim_sup=lim_sup)
            #  df_recruitment.loc[cell_mask & bout_mask, 'mean_spks'] = get_mean_spks(Exp, df_bout, bout, Cell,
            #                                                                        lim_inf=lim_inf, lim_sup=lim_sup)

            del bout_mask

        #  TODO: do this outside, in the functio nget_spks_during_no_mov directly, that changes Cell attribute and
        # TODO: function spks_during_no_mov should return updated df recruitment
        # df_recruitment.at[cell_mask, 'spks_no_mov'] = get_spks_during_no_mov(Cell, Exp.ta_resampled)
        #  df_recruitment.at[cell_mask, 'spks_during_forward'] = get_all_spks_during_forward(Exp, df_bout, Cell)

        del cell_mask
        print('Processed {} cells out of {}'.format(i, len(Cells)))

    return df_recruitment


def getStatsActivityDuringBout(df_recruitment, bout_type, Exp, MASKS):
    mask = MASKS[bout_type]
    for i, plane in enumerate(Exp.suite2pData.keys()):
        for cell in df_recruitment[df_recruitment.plane == plane].cell_id.unique():
            cell_mask = df_recruitment.cell_id == cell
            a = df_recruitment.loc[cell_mask & mask, 'recruitment']
            b = df_recruitment.loc[cell_mask & mask, 'norm_max_dff']
            df_recruitment.loc[cell_mask, 'mean_recruitment_f'] = np.nanmean(a)
            df_recruitment.loc[cell_mask, 'median_recruitment_f'] = np.nanmean(a)
            df_recruitment.loc[cell_mask, 'mean_max_dff_f'] = np.nanmean(b)
            df_recruitment.loc[cell_mask, 'median_max_dff_f'] = np.nanmean(b)
        print('Processed {} planes out of {}'.format(i, len(Exp.suite2pData.keys())))
    return df_recruitment


def set_spks_dif_during_left_or_right_turns(Cell, df_recruitment, Exp, df_bout, turns_mask, p0=0.05):
    # TODO: add in df_recruitment the spks for no movement
    # TODO: add in df_recruitment the spks for all forward in one column

    cell_id = Cell.cellID
    x = get_all_spks_during_all_bouts(Exp, df_bout[turns_mask & (df_bout.Max_Bend_Amp > 0)], Cell)
    y = get_all_spks_during_all_bouts(Exp, df_bout[turns_mask & (df_bout.Max_Bend_Amp < 0)], Cell)

    #  get if cell spks are greater during forward than during no mov
    _, pvalue_greater = ranksums(x, y, alternative='greater')
    setattr(Cell, 'more_active_during_left_than_right', pvalue_greater < p0)
    df_recruitment.loc[df_recruitment.cell_id == cell_id, 'more_active_during_left_than_right'] = pvalue_greater < p0

    return df_recruitment


def more_active_during_locomotion(Cells, Exp, df_bout, df_recruitment, dict_bout_types, p0=0.05):
    # Computes whether each cell is more active during one type of bouts than when the fish is not moving.
    # TO do so, compares the distribution of A: spike rates during bout of a given type and B: spike rates during
    # resting periods (no movement detected). Returns True if A significantly greater than B in ranksum test.
    for Cell in Cells:
        y = get_spks_during_no_mov(Cell, Exp.ta_resampled)
        for bout_type, mask in dict_bout_types.items():
            df_bout_short = df_bout[mask].copy()

            if df_bout_short.empty:
                coef = np.nan
                output = np.nan

            else:
                x = get_all_spks_during_all_bouts(Exp, df_bout_short, Cell)
                coef, pvalue = ranksums(x, y, alternative='greater')
                if pvalue < p0:
                    output = True
                else:
                    output = False
                    coef = np.nan
            df_recruitment.loc[
                (df_recruitment.cell_id == Cell.cellID), 'more_active_during_bout_type_{}'.format(bout_type)] = output
            df_recruitment.loc[
                (df_recruitment.cell_id == Cell.cellID), 'coef_more_active_during_bout_type_{}'.format(
                    bout_type)] = coef
            setattr(Cell, 'more_active_during_bout_type_{}'.format(bout_type), output)
            setattr(Cell, 'coef_more_active_during_bout_type_{}'.format(bout_type), coef)

    return df_recruitment


def get_all_bouts_combinations(dict_bout_types):
    a = list(itertools.combinations(dict_bout_types.keys(), 2))
    output = a + [tuple(reversed(i)) for i in a]
    return output


def more_active_during_bout_types_reciprocal(Cells, Exp, df_bout, df_recruitment, dict_bout_types, p0=0.05):
    all_combinations = get_all_bouts_combinations(dict_bout_types)
    for Cell in Cells:
        for bout_type1, bout_type2 in all_combinations:

            if any([df_bout[dict_bout_types[bout_type1]].empty, df_bout[dict_bout_types[bout_type2]].empty]):
                coef = np.nan
                output = np.nan

            else:
                x = get_all_spks_during_all_bouts(Exp, df_bout[dict_bout_types[bout_type1]], Cell)
                y = get_all_spks_during_all_bouts(Exp, df_bout[dict_bout_types[bout_type2]], Cell)
                # test if x greater than y
                coef, pvalue = ranksums(x, y, alternative='greater')
                if pvalue < p0:
                    output = True
                else:
                    output = False
                    coef = np.nan
            df_recruitment.loc[
                (df_recruitment.cell_id == Cell.cellID),
                'more_active_during_{}_than_during_{}'.format(bout_type1,
                                                              bout_type2)] = output
            df_recruitment.loc[
                (df_recruitment.cell_id == Cell.cellID),
                'coef_more_active_during_{}_than_during_{}'.format(bout_type1,
                                                                   bout_type2)] = coef

            setattr(Cell, 'more_active_during_{}_than_during_{}'.format(bout_type1,
                                                                        bout_type2), output)
            setattr(Cell, 'coef_more_active_during_{}_than_during_{}'.format(bout_type1,
                                                                             bout_type2), coef)

    return df_recruitment


def get_list_cells_active_during_bout_types(df_recruitment):
    f_spe = df_recruitment[df_recruitment.more_active_during_bout_type_forward].cell_id.unique()
    l_spe = df_recruitment[df_recruitment.more_active_during_bout_type_left_turns].cell_id.unique()
    r_spe = df_recruitment[df_recruitment.more_active_during_bout_type_right_turns].cell_id.unique()
    return f_spe, l_spe, r_spe


## Plotting

def plot_ex_forward_cells(Cells, Exp, df_bout, df_recruitment, savefig=False):
    # TODO: add periods of no mov to grey shade in the graph
    #  TODO: add periods of forward mov to blue shade in the graph

    cell_forward_greater = list(df_recruitment[df_recruitment.more_active_during_bout_type_forward].cell_id.unique())

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Distribution of spike rates during forward bouts vs no movement frames (log scale)')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x1 = Exp.tail_angle.copy()
    x2 = Exp.ta_pure_forward.copy()
    ax1.plot(Exp.time_indices_bh, zscore(x1), color='grey', label='zscore ta trace ')
    ax1.plot(Exp.time_indices_bh, zscore(x2), color='royalblue', alpha=0.5,
             label='zscore ta trace only for forward bouts')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Tail angle (/10) [°] / z score DFF')
    fig2, ax2 = plt.subplots(1, 2)
    fig2.suptitle('Position of cells')
    ax2[0].imshow(Exp.mean_background, cmap='Greys')
    sns.scatterplot(data=df_recruitment.drop_duplicates('cell_id'), x='x_pos', y='norm_y_pos', hue='bout_id',
                    cmap='Greys', ax=ax2[0], alpha=0.5)
    ax2[1].imshow(Exp.mean_background, cmap='Greys')
    sns.scatterplot(data=df_recruitment.drop_duplicates('cell_id'), x='x_pos', y='plane', hue='bout_id', cmap='Greys',
                    ax=ax2[1], alpha=0.5)

    for i, cell in enumerate(sample(cell_forward_greater, 9)):
        x = get_all_spks_during_forward(Exp, df_bout, Cells[cell])
        y = get_spks_during_no_mov(Cells[cell], Exp.ta_resampled)
        ax.flatten()[i].hist(x, density=True, label='during_forward', alpha=0.5)
        ax.flatten()[i].hist(y, density=True, label='during_quiet', alpha=0.5)
        ax.flatten()[i].set_title('Cell {}'.format(cell))
        ax.flatten()[i].set_yscale("log")
        ax.flatten()[i].legend()

        y = Cells[cell].dff_corrected.copy()
        ax1.plot(Exp.time_indices_SCAPE, zscore(y, nan_policy='omit') + (i + 1) * 7,
                 label='zscore dff corrected cell{}'.format(cell))
        y = np.array(pd.Series(Cells[cell].spks.copy()).rolling(3, center=True).mean())
        ax1.plot(Exp.time_indices_SCAPE, y + (i + 1) * 7, alpha=0.7, color='grey',
                 label='spks corrected cell{}'.format(cell))
        ax1.legend()
        ax2[0].plot(Cells[cell].x_pos, Cells[cell].norm_y_pos, 'o', label='cell_{}'.format(cell))
        ax2[1].plot(Cells[cell].x_pos, Cells[cell].plane, 'o', label='cell_{}'.format(cell))
        plt.legend()
    ax.flatten()[i].set_xlabel('Spike rate [UA]')
    ax.flatten()[i].set_ylabel('Density')
    ax1.set_xlim(70, 150)

    if savefig:
        fig.savefig(Exp.fig_path + '/example_cells_great_spikes_during_forward_distrib2.svg')
        fig1.savefig(Exp.fig_path + '/example_cells_great_spikes_during_forward_traces2.svg')
        fig2.savefig(Exp.fig_path + '/example_cells_great_spikes_during_forward_map2.svg')


def map_forward_cells_per_plane(Exp, df_recruitment, savefig=False, coef_hue=False):
    df_short = df_recruitment[df_recruitment.bout_id == 0].copy()

    fig, ax = plt.subplots(6, 5, figsize=(15, 20))
    fig.suptitle('Purple: cells whose spike rate is sig greater during forward than when fish not moving')
    planes = list(Exp.suite2pData.keys())
    planes.sort()
    for i, plane in enumerate(planes):
        ax.flatten()[i].imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys')
        if not coef_hue:

            sns.scatterplot(data=df_short[(df_short.plane == plane) &
                                          (~df_short.more_active_during_bout_type_forward)],
                            x='x_pos', y='y_pos',
                            hue='more_active_during_bout_type_forward',
                            palette='Greys',
                            ax=ax.flatten()[i])
            sns.scatterplot(data=df_short[(df_short.plane == plane) &
                                          (df_short.eval('more_active_during_bout_type_forward'))],
                            x='x_pos', y='y_pos',
                            hue='more_active_during_bout_type_forward', palette='Reds',
                            ax=ax.flatten()[i])

        else:

            sns.scatterplot(data=df_short[(df_short.plane == plane) &
                                          (~df_short.more_active_during_bout_type_forward)],
                            x='x_pos', y='y_pos',
                            hue='more_active_during_bout_type_forward',
                            palette='Greys',
                            ax=ax.flatten()[i])
            sns.scatterplot(data=df_short[(df_short.plane == plane) &
                                          (df_short.more_active_during_bout_type_forward)],
                            x='x_pos', y='y_pos',
                            hue='coef_more_active_during_bout_type_forward',
                            hue_norm=(np.nanmin(df_short.coef_more_active_during_bout_type_forward),
                                      np.nanmax(df_short.coef_more_active_during_bout_type_forward)),
                            ax=ax.flatten()[i])
    plt.tight_layout()
    if savefig:
        plt.savefig(Exp.fig_path + '/map_cells_greater_spikes_during_forward.svg')


def map_recruitment_single_bout_type(Exp, df_recruitment, bout_type, savefig=False, coef_hue=False):
    min_bout_id = df_recruitment.bout_id.min()
    df_short = df_recruitment[df_recruitment.bout_id == min_bout_id].copy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 20))
    fig.suptitle('Cells whose spike rate is sig greater during {} than when fish not moving'.format(bout_type))
    ax[0].set_title('Dorsal view')
    ax[0].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
    sns.scatterplot(data=df_short[df_short['more_active_during_bout_type_{}'.format(bout_type)] == False],
                    x='x_pos', y='norm_y_pos',
                    hue='more_active_during_bout_type_{}'.format(bout_type),
                    palette='Greys',
                    alpha=0.35,
                    ax=ax[0])
    ax[1].set_title('Sagittal view')
    ax[1].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
    sns.scatterplot(data=df_short[df_short['more_active_during_bout_type_{}'.format(bout_type)] == False],
                    x='x_pos', y='plane',
                    hue='more_active_during_bout_type_{}'.format(bout_type),
                    palette='Greys',
                    alpha=0.35,
                    ax=ax[1])

    if not coef_hue:
        hue_code = 'more_active_during_bout_type_{}'.format(bout_type)
    else:
        hue_code = 'coef_more_active_during_bout_type_{}'.format(bout_type)

    sns.scatterplot(data=df_short, x='x_pos', y='norm_y_pos',
                    hue=hue_code,
                    alpha=0.5,
                    ax=ax[0])
    sns.scatterplot(data=df_short, x='x_pos', y='plane',
                    hue=hue_code,
                    alpha=0.5,
                    ax=ax[1])

    plt.tight_layout()
    if savefig:
        plt.savefig(Exp.fig_path + '/map_cells_greater_spikes_during_{}.svg'.format(bout_type))


def map_all_recruitment_comparisons(Exp, df_recruitment, savefig=False):

    df_short = df_recruitment[df_recruitment.bout_id == 0].copy()
    columns_interest = [i for i in df_short.columns.values.tolist() if i.startswith('more_active_during')]
    for col in columns_interest:
        fig, ax = plt.subplots(1, 2, figsize=(15, 20))
        fig.suptitle('Purple: cells {}'.format(col))
        ax[0].set_title('Dorsal view')
        ax[0].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short[df_short[col] == False],
                        x='x_pos', y='norm_y_pos',
                        hue=col,
                        palette='Greys',
                        alpha=0.35,
                        ax=ax[0])
        ax[1].set_title('Sagittal view')
        ax[1].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short[df_short[col] == False],
                        x='x_pos', y='plane',
                        hue=col,
                        palette='Greys',
                        alpha=0.35,
                        ax=ax[1])

        sns.scatterplot(data=df_short[df_short[col] == True], x='x_pos', y='norm_y_pos',
                        hue=col,
                        alpha=0.5,
                        ax=ax[0])
        sns.scatterplot(data=df_short[df_short[col] == True], x='x_pos', y='plane',
                        hue=col,
                        alpha=0.5,
                        ax=ax[1])

        plt.tight_layout()
        if savefig:
            plt.savefig(Exp.fig_path + '/map_{}.svg'.format(col))


def map_activation_during_bouts_types(Exp,
                                      df_recruitment,
                                      palettes=None,
                                      savefig=False):
    if palettes is None:
        palettes = {'forward': ["#BEBEBE", "#372248"],
                    'left_turns': ["#BEBEBE", "#F46036"],
                    'right_turns': ["#BEBEBE", "#006E90"]}

    df_short = df_recruitment.drop_duplicates('cell_id').copy()
    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle(
        'Cells active during bout types\nn={} fish, {} recording, {} bouts'.format(len(df_recruitment.fishID.unique()),
                                                                                   len(df_recruitment.runID.unique()),
                                                                                   len(
                                                                                       df_recruitment.bout_id.unique())))
    for i, key in enumerate(palettes.keys()):
        ax[i, 0].set_title('Dorsal view')
        ax[i, 0].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short, x='x_pos', y='norm_y_pos',
                        hue='more_active_during_bout_type_{}'.format(key),
                        alpha=0.5,
                        palette=sns.color_palette(palettes[key]),
                        ax=ax[i, 0])
        ax[i, 1].set_title('Sagittal view')
        ax[i, 1].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short, x='x_pos', y='plane',
                        hue='more_active_during_bout_type_{}'.format(key),
                        alpha=0.5,
                        palette=sns.color_palette(palettes[key]),
                        ax=ax[i, 1])
    if savefig:
        plt.savefig(Exp.fig_path + '/active_during_bout_types.svg')


def map_cross_activation_during_bouts_types(Exp,
                                            df_recruitment,
                                            palettes=None,
                                            savefig=False):
    if palettes is None:
        palettes = {'F+L': ["#987284"],
                    'F+R': ["#75B9BE"],
                    'L+R': ["#519E8A"],
                    'F+L+R': ["#000000"]}

    f_spe, l_spe, r_spe = get_list_cells_active_during_bout_types(df_recruitment)

    combinations = {'F+L': set.intersection(*map(set, [f_spe, l_spe])),
                    'F+R': set.intersection(*map(set, [f_spe, r_spe])),
                    'L+R': set.intersection(*map(set, [l_spe, r_spe])),
                    'F+L+R': set.intersection(*map(set, [f_spe, l_spe, r_spe]))}

    df_short = df_recruitment.drop_duplicates('cell_id').copy()
    fig, ax = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle('Cells crpss-active during bout types')

    for i, combination in enumerate(combinations.keys()):
        cells = combinations[combination]
        ax[i, 0].set_title('{}\nDorsal view'.format(combination))
        ax[i, 0].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short[~df_short.cell_id.isin(cells)], x='x_pos', y='norm_y_pos',
                        hue='bout_id',
                        alpha=0.25,
                        palette='Greys',
                        ax=ax[i, 0])
        sns.scatterplot(data=df_short[df_short.cell_id.isin(cells)], x='x_pos', y='norm_y_pos',
                        hue='bout_id',
                        alpha=0.5,
                        palette=sns.color_palette(palettes[combination]),
                        ax=ax[i, 0])

        ax[i, 1].set_title('Sagittal view')
        ax[i, 1].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short[~df_short.cell_id.isin(cells)], x='x_pos', y='plane',
                        hue='bout_id',
                        alpha=0.25,
                        palette='Greys',
                        ax=ax[i, 1])
        sns.scatterplot(data=df_short[df_short.cell_id.isin(cells)], x='x_pos', y='plane',
                        hue='bout_id',
                        alpha=0.5,
                        palette=sns.color_palette(palettes[combination]),
                        ax=ax[i, 1])

    if savefig:
        plt.savefig(Exp.fig_path + '/map_cross_active_during_bout_types.svg')


def map_cross_activation_during_bouts_types(Exp,
                                            df_recruitment,
                                            palettes=None,
                                            savefig=False):
    if palettes is None:
        palettes = {'L-F': ["#987284"],
                    'R-F': ["#75B9BE"]}

    f_spe, l_spe, r_spe = get_list_cells_active_during_bout_types(df_recruitment)

    combinations = {'L-F': set.difference(*map(set, [l_spe, f_spe])),
                    'R-F': set.difference(*map(set, [r_spe, f_spe]))}

    df_short = df_recruitment.drop_duplicates('cell_id').copy()
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle('Cells diff-active during bout types')

    for i, combination in enumerate(combinations.keys()):
        cells = combinations[combination]
        ax[i, 0].set_title('{}\nDorsal view'.format(combination))
        ax[i, 0].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short[~df_short.cell_id.isin(cells)], x='x_pos', y='norm_y_pos',
                        hue='bout_id',
                        alpha=0.25,
                        palette='Greys',
                        ax=ax[i, 0])
        sns.scatterplot(data=df_short[df_short.cell_id.isin(cells)], x='x_pos', y='norm_y_pos',
                        hue='bout_id',
                        alpha=0.5,
                        palette=sns.color_palette(palettes[combination]),
                        ax=ax[i, 0])

        ax[i, 1].set_title('Sagittal view')
        ax[i, 1].imshow(Exp.mean_background[:, 0:495], cmap='Greys')
        sns.scatterplot(data=df_short[~df_short.cell_id.isin(cells)], x='x_pos', y='plane',
                        hue='bout_id',
                        alpha=0.25,
                        palette='Greys',
                        ax=ax[i, 1])
        sns.scatterplot(data=df_short[df_short.cell_id.isin(cells)], x='x_pos', y='plane',
                        hue='bout_id',
                        alpha=0.5,
                        palette=sns.color_palette(palettes[combination]),
                        ax=ax[i, 1])

    if savefig:
        plt.savefig(Exp.fig_path + '/map_dif_active_during_bout_types.svg')


def map_cross_activation_plane_wise(Exp, Cells, df_recruitment, savefig=False):
    f_spe = df_recruitment[df_recruitment.more_active_during_bout_type_forward].cell_id.unique()
    l_spe = df_recruitment[df_recruitment.more_active_during_bout_type_left_turns].cell_id.unique()
    r_spe = df_recruitment[df_recruitment.more_active_during_bout_type_right_turns].cell_id.unique()

    fig, ax = plt.subplots(6, 5, figsize=(15, 20))
    fig.suptitle('Active during F, L and R')
    planes = list(Exp.suite2pData.keys())
    planes.sort()
    for i, plane in enumerate(planes):
        ax.flatten()[i].imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys')
    for cell in set(f_spe).intersection(l_spe).intersection(r_spe):
        plane = Cells[cell].plane
        subplot = np.where(np.array(planes) == plane)[0][0]
        ax.flatten()[subplot].plot(Cells[cell].x_pos, Cells[cell].y_pos, 'o', color='green', alpha=0.6)
    plt.tight_layout()
    if savefig:
        plt.savefig(Exp.fig_path + '/map_active_f_l_r.svg')

    fig, ax = plt.subplots(6, 5, figsize=(15, 20))
    fig.suptitle('Cells active during F and L')
    planes = list(Exp.suite2pData.keys())
    planes.sort()
    for i, plane in enumerate(planes):
        ax.flatten()[i].imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys')
    for cell in set(f_spe).intersection(l_spe):
        plane = Cells[cell].plane
        subplot = np.where(np.array(planes) == plane)[0][0]
        ax.flatten()[subplot].plot(Cells[cell].x_pos, Cells[cell].y_pos, 'o', color='purple', alpha=0.6)
    plt.tight_layout()
    if savefig:
        plt.savefig(Exp.fig_path + '/map_active_f_l.svg')

    fig, ax = plt.subplots(6, 5, figsize=(15, 20))
    fig.suptitle('Cells active during F and R')
    planes = list(Exp.suite2pData.keys())
    planes.sort()
    for i, plane in enumerate(planes):
        ax.flatten()[i].imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys')
    for cell in set(f_spe).intersection(r_spe):
        plane = Cells[cell].plane
        subplot = np.where(np.array(planes) == plane)[0][0]
        ax.flatten()[subplot].plot(Cells[cell].x_pos, Cells[cell].y_pos, 'o', color='royalblue', alpha=0.6)
    plt.tight_layout()
    if savefig:
        plt.savefig(Exp.fig_path + '/map_active_f_r.svg')

    fig, ax = plt.subplots(6, 5, figsize=(15, 20))
    fig.suptitle('Cells active during L and R:\nblack -> ONLY!\norange -> also during F')
    planes = list(Exp.suite2pData.keys())
    planes.sort()
    for i, plane in enumerate(planes):
        ax.flatten()[i].imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys')
    for cell in set(l_spe).intersection(r_spe):
        plane = Cells[cell].plane
        subplot = np.where(np.array(planes) == plane)[0][0]
        if cell in f_spe:
            color = 'orange'
        else:
            color = 'black'
        ax.flatten()[subplot].plot(Cells[cell].x_pos, Cells[cell].y_pos, 'o', color=color, alpha=0.6)
    plt.tight_layout()
    if savefig:
        plt.savefig(Exp.fig_path + '/map_active_l_r.svg')


def plot_spks_vs_kinematic(Cells, Exp, df_bout, savefig=False):
    cell_forward_greater = [Cell.cellID for Cell in Cells if Cell.more_active_during_forward]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i, cell in enumerate(cell_forward_greater[:10]):
        spks = get_mean_spks_during_bouts(Exp, df_bout, Cells[cell])
        ax[0, 0].scatter(spks, df_bout.Max_Bend_Amp, label=str(cell), alpha=0.5)
        ax[0, 0].set_xlabel('Spike rate [UA]')
        ax[0, 0].set_ylabel('Abs Max Bend Amplitude')
        ax[0, 1].scatter(spks, df_bout.median_bend_amp, label=str(cell), alpha=0.5)
        ax[0, 1].set_xlabel('Spike rate [UA]')
        ax[0, 1].set_ylabel('Median Bend Amplitude')
        ax[0, 1].set_ylim((-20, 20))
        ax[1, 0].scatter(spks, df_bout.median_iTBF, label=str(cell), alpha=0.5)
        ax[1, 0].set_xlabel('Spike rate [UA]')
        ax[1, 0].set_ylabel('Median iTBF [Hz]')
        ax[1, 1].scatter(spks, df_bout.mean_TBF, label=str(cell), alpha=0.5)
        ax[1, 1].set_xlabel('Spike rate [UA]')
        ax[1, 1].set_ylabel('Mean iTBF [Hz]')
        plt.tight_layout()
        if savefig:
            plt.savefig(Exp.fig_path + '/spks_vs_kinematics_cell{}.svg'.format(cell))


def plot_maps_recruitment_bout_type(plane, colorVariable, MASKS, Exp, df_noDupl, ax):
    for i, bout_type in enumerate(MASKS.keys()):
        ax[i].imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys', vmax=100)
        sns.scatterplot(data=df_noDupl[df_noDupl.plane == plane],
                        y='y_pos', x='x_pos', hue=colorVariable + '_' + bout_type, ax=ax[i])
