import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from suite2p.extraction import dcnv


def correct_suite2p_outputs(F, iscell, Fneu, neu_factor=0.7):
    nROIs, nFrames = F.shape
    print('Number of ROIs: ', nROIs)
    print('Number of frames: ', nFrames)
    cells_index = np.flatnonzero(iscell[:, 0])
    nCells = len(cells_index)
    print('Number of cells: ', nCells)
    F_corrected = np.ones(F.shape)
    # correction based on recommandation of suite2p, correction made with the neuropile fluorescence
    for ROI in range(F.shape[0]):
        F_corrected[ROI,] = F[ROI,] - neu_factor * Fneu[ROI,]
    # if you don't want motion artifact correction automated, comment the next line
    print('Calculating F corrected and cells index')
    return F_corrected, cells_index


def calc_dff(cells, f_corrected, bad_frames=None, baseline=None):

    if bad_frames is not None:
        f_corrected[:, bad_frames] = np.nan

    output = np.zeros(f_corrected.shape)
    output[:] = np.nan
    output_noise = [np.nan]*f_corrected.shape[0]

    if baseline is None:
        for cell in cells:
            baseline = np.nanmedian(np.nanpercentile(f_corrected[cell,], 3))
            output[cell,] = [f_corrected[cell, t] - baseline / baseline for t in range(f_corrected.shape[1])]
            output_noise[cell] = np.std(baseline)

    else:
        for cell in cells:
            baseline = np.nanmedian(f_corrected[cell, baseline[0]:baseline[1]])
            output[cell,] = [f_corrected[cell, t] - baseline / baseline for t in range(f_corrected.shape[1])]
            output_noise[cell] = np.nanstd(baseline)

    return output, output_noise


def remove_motion_artifact(traces, Exp):
    traces[:, Exp.bad_frames] = np.nan
    output = np.array(pd.DataFrame(traces).interpolate(axis=1))
    return output


def runSpksExtraction(f, Exp, batch_size, tau, fs):
    """

    :param f: ndarray of floats, nCells x nTimePoints
    :param ops: dict, params for deconvolution algorithm
    :return: ndarray of ints, nCells x nTimePoints, approximated spike rate at each time step for each cell

    """
    f = remove_motion_artifact(f, Exp)
    output = dcnv.oasis(F=f, batch_size=batch_size, tau=tau, fs=fs)
    return output
