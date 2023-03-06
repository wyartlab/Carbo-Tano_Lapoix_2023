import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore

import utils.recruitmentCellBout as rct
import utils.import_data as load
import utils.createExpClass as createExpClass
import utils.createCellClass as createCellClass


def calc_dff(cells, f_corrected, bad_frames=None, baseline=None):

    if bad_frames is not None:
        f_corrected[:, bad_frames] = np.nan

    output = np.zeros(f_corrected.shape)
    output[:] = np.nan

    if baseline is None:
        for cell in cells:
            baseline = np.nanmedian(np.nanpercentile(f_corrected[cell,], 3))
            output[cell,] = [f_corrected[cell, t] - baseline / baseline for t in range(f_corrected.shape[1])]

    else:
        for cell in cells:
            baseline = np.nanmedian(f_corrected[cell, baseline[0]:baseline[1]])
            output[cell,] = [f_corrected[cell, t] - baseline / baseline for t in range(f_corrected.shape[1])]

    return output


### Initialise

summary_csv = load.load_summary_csv(
    'https://docs.google.com/spreadsheets/d/1VHFmX8j8rfwDiKghT5tb0LxZfmB5_qrgqYmcJxmB7RE/edit#gid=1097839266')
exp_id = 79
Exp = createExpClass.Run(summary_csv, exp_id)
bad_frames = [int(i) for i in summary_csv.bad_frames[exp_id].split(',')]

### Load pre-processed information

df_frame, df_bout = load.load_behavior_dataframe(Exp)
ta = np.array(df_frame.Tail_angle)
time_indices_bh = np.array(df_frame.Time_index)

Exp.load_suite2p_outputs()

all_dff = {}
all_dff_zscore = {}
all_cells = {}

for plane in Exp.suite2pData.keys():
    Exp.correct_suite2p_outputs(plane)
    Exp.filter_f(plane)
    cells = Exp.suite2pData[plane]['cells']
    # f_corrected = Exp.suite2pData[plane]['F_corrected_filter']
    f_corrected = Exp.suite2pData[plane]['F_corrected']
    dff = calc_dff(cells, f_corrected, bad_frames=bad_frames)
    all_dff[plane] = dff
    all_cells[plane] = cells
    zscore_dff = np.zeros(dff.shape)
    for cell in cells:
        zscore_dff[cell,] = zscore(dff[cell,], nan_policy='omit')
    all_dff_zscore[plane] = zscore_dff

print('Total number of cells: {}'.format(sum([len(all_cells[key]) for key in all_cells.keys()])))
    # suite2p_outputs[plane]['DFF'], _ = calc_dff(suite2p_outputs[plane]['F_corrected'], suite2p_outputs[plane]['cells'], fps_ci)

time_indices_ci = np.arange(dff.shape[1]) / Exp.frameRateSCAPE

# chunk

frames_bh = (136513, 142515)
frames_scape = (2528, 2639)


def get_cells_interest(dff, frames_scape, nCellsToTake=10):
    """
    During period of recording specified by user input, output the N cells the most active.

    :param dff: array of shape (nRois, nFrames)
    :param frames_scape: tuple, (limit_inf, limit_sup) of the given period of interest
    :param nCellsToTake: int, number of cells to sort
    :return: array
    """
    short_dff = dff[:,frames_scape[0]:frames_scape[1]]
    sum_dff = np.nanmax(short_dff, axis=1)
    cells_interest = np.argsort(sum_dff)[-nCellsToTake:]

    return cells_interest


def plot_cells_pos(cells, Exp, ax):
    for cell in cells:
        y, x = Exp.suite2pData[plane]['stat'][cell]['med'][0], Exp.suite2pData[plane]['stat'][cell]['med'][1]
        ax.plot([x], [y], 'o')
        ax.annotate(cell, (int(x), int(y)), color='coral')


def plot_cells_trace(cells, dff, time_indices, ax):
    for i, cell in enumerate(cells):

        cell_trace = dff[cell,]
        ax.plot(time_indices,
                cell_trace + i * 1.5,
                label='cell_{}'.format(cell))


def plot_example_traces_plane(plane, dff, frames_bh, frames_scape, Exp,
                              nCellsToTake=10, zscore_trace=False):

    cells_interest = get_cells_interest(dff, frames_scape, nCellsToTake)

    fig, ax = plt.subplots(1, 2, num=plane)
    fig.suptitle('Cells the most active during period of time for plane {}'.format(plane))

    ax[0].imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys')
    ax[0].set_title('Cells position')
    plot_cells_pos(cells_interest, Exp, ax[0])

    if zscore_trace:
        ax[1].set_title('zscore dff traces')
    else:
        ax[1].set_title('dff traces')

    plot_cells_trace(cells_interest,
                     dff[:,frames_scape[0]:frames_scape[1]],
                     time_indices_ci[frames_scape[0]:frames_scape[1]],
                     ax[1])

    if zscore_trace:
        factor = 5
    else:
        factor = 40
    ax[1].plot(time_indices_bh[frames_bh[0]:frames_bh[1]],
               ta[frames_bh[0]:frames_bh[1]]/20-factor,
               color='silver')
    return fig


for plane in Exp.suite2pData.keys():
    zscore_dff = all_dff_zscore[plane]
    fig = plot_example_traces_plane(plane, zscore_dff, frames_bh, frames_scape, Exp, nCellsToTake=10, zscore_trace=True)
    fig.savefig('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/Illustrations/dff_ta/220210_F2_run5/'+str(plane)+'.svg')
