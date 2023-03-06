import numpy as np
import pandas as pd
import math
import os
from glob import glob
import json
import matplotlib.pyplot as plt
from scipy.integrate import simps


def load_zz_results(trial_path):
    try:
        filepath = glob(trial_path + '/results*')[0]
    except (FileNotFoundError, NotADirectoryError, IndexError):
        print('No ZZ output found at this path:', trial_path)
        filepath = False
    return filepath


def load_ZZ_output(trial_path):
    filename = load_zz_results(trial_path)
    if filename:
        with open(filename) as f:
            supstruct = json.load(f)
    else:
        supstruct = None
    return supstruct


# function to return a list of index corresponding to element in a list (lst) filling a condition
def find_indices(lst, condition):
    """return the indices of elements in lst fitting the condition"""
    return [i for i, elem in enumerate(lst) if condition(elem)]


def bout_num(frame_i, structure, NBout):
    """returns the bout number that happened at the frame indexed frame_i
    If no bout was happening at this frame, returns NaN
    structure is the ZZoutput already imported from txt format"""
    output = np.nan
    numBout = 0
    while math.isnan(output) is True:
        if numBout == NBout:
            break
        else:
            indices = range(get_bout_start(numBout, structure), get_bout_end(numBout, structure) + 1)
            if frame_i in indices:
                output = int(numBout)
            else:
                numBout += 1
    return output


def relative_index(frame_i, structure, NBout):
    """returns the relative frame index of the bout: was the frame the first one, the second... of the bout
    if the frame does not correspond to a bout, returns NaN
    structure is the ZZoutput already imported from txt format"""
    if math.isnan(bout_num(frame_i, structure, NBout)) is True:
        output = np.nan
    else:
        output = int(frame_i - get_bout_start(bout_num(frame_i, structure, NBout), structure))
    return output


# raises an error when the frame does not correspond to a bout (bout_num = NaN)
# so if the Type Error is raised, returns NaN value
# if not, returns the angle corresponding to the relative index of the frame in the bout
def tail_angle(frame_i, structure, NBout):
    """"returns tail_angle measured by ZZ, extracted from structure, at frame_i
    If the frame does not correspond to a bout, returns NaN"""
    numBout = bout_num(frame_i, structure, NBout)
    try:
        output = 57.2958 * structure[numBout]["TailAngle_smoothed"][relative_index(frame_i, structure, NBout)]
    except TypeError:
        output = np.nan
    return output


# if the frame corresponds to a peak, return the absolute amplitude of the peak
# returns NaN otherwise
def bend_amplitude(frame_i, structure, NBout):
    """For frame_i, returns the amplitude of the tail angle if the frame corresponds to a bend detected by ZZ
    Information extracted from structure"""
    numBout = bout_num(frame_i, structure, NBout)
    if math.isnan(numBout) is True:
        output = np.nan
    elif frame_i in bend_indices(numBout, structure):
        bend_index = find_indices(bend_indices(numBout, structure), lambda e: e == frame_i)[0]
        # convert from radians to angles
        output = structure[numBout]["Bend_Amplitude"][bend_index] * 57.2958
    else:
        output = np.nan
    return output


# return bend relative index in the bout
def bend_index(frame_i, structure, NBout):
    """For frame_i, returns the bend relative index in the bout if the frame corresponds to a bend detected by ZZ
    structure is the ZZoutput already imported from txt format"""
    numBout = bout_num(frame_i, structure, NBout)
    output = np.nan
    if math.isnan(numBout) is False and frame_i in bend_indices(numBout, structure):
        output = int(find_indices(bend_indices(numBout, structure), lambda e: e == frame_i)[0])
    return output


# return instantaneous TBF at eack peak, starting from the second bend of the bout
def instant_TBF(frame_i, structure, NBout, dataset, fq):
    """ Returns calculated instantaneous TBF at each bend frame indices from structure
    Extrapolated by calculating the frequency of bend based on the time between the previous bend and the actual one

    :param frame_i: int, frame at which you want to compute eventual tbf
    :param structure: list, ZZ output structure for a given fish that has N dictionnaries where N is the number of bouts detected.
    :param NBout: number of bouts detected
    :param dataset: pandas dataframe with infos on each frame of the recording.
    :param fq: float, frame rate of acquisition
    :return: if any bend happened at this frame, return the invert period it took from previous bend * 2 (as a complete cycle is made of 2 bends).
    """
    output = np.nan
    if math.isnan(dataset.Bend_Amplitude[frame_i]) is False and bend_index(frame_i, structure, NBout) != 0:
        numBout = bout_num(frame_i, structure, NBout)
        frame_last_bend = structure[numBout]["Bend_TimingAbsolute"][bend_index(frame_i, structure, NBout) - 1]
        output = 1 / ((frame_i - frame_last_bend) * 2 / fq)
    return output


def get_time(frame_i, fq):
    """returns the time corresponding to frame_i
    Based on fq, which by default is 250"""
    return frame_i / fq


# returns the distance between the tip of the tail and the bladder(supposed fixed) at one frame
def tail_bladder_d(frame_i, structure, NBout):
    """For frame_i, calculates the distance between tip of the tail and bladder, from structure"""
    numBout = bout_num(frame_i, structure, NBout)
    if math.isnan(numBout) is True:
        output = np.nan
    else:
        x_t = structure[numBout]["TailX_VideoReferential"][frame_i - get_bout_start(numBout, structure)][0]
        x_b = structure[numBout]["TailX_VideoReferential"][frame_i - get_bout_start(numBout, structure)][1]
        y_t = structure[numBout]["TailY_VideoReferential"][frame_i - get_bout_start(numBout, structure)][0]
        y_b = structure[numBout]["TailY_VideoReferential"][frame_i - get_bout_start(numBout, structure)][1]
        output = math.sqrt((x_t - x_b) ** 2 + (y_t - y_b) ** 2)
    return output


# Find number of oscillations in a bout as the number of bends divided by 2
# i don't know this if condition is
def N_osc_b(numBout, structure):
    """Returns the number of oscillations that happened in numBout, from structure"""
    output = len(structure[numBout]["Bend_Amplitude"]) / 2
    if output == 0:
        output = 1
    return output


# Find bout duration as the number of frames composing it divided by frame rate
def bout_duration(numBout, structure, fq):
    """returns bout duration of numBout from structure, based on frequency fq, which by default is 250"""
    return (get_bout_end(numBout, structure) - get_bout_start(numBout, structure)) / fq


# Bout Start and Bout End
def get_bout_start(numBout, structure):
    """returns the starting frame of the bout number numBout, stored in structure """
    return int(structure[numBout]["BoutStart"])


def get_bout_end(numBout, structure):
    """returns the ending frame of the bout number numBout, stored in structure """
    return int(structure[numBout]["BoutEnd"])


# Find Maximal Bend Amplitude of a given bout
def max_bend_amp(numBout, structure):
    """returns the maximal bend amplitude found in the bout number numBout, stored in structure """
    bend_series = pd.Series(structure[numBout]["Bend_Amplitude"])
    try:
        # add an argument to the max function to look for the maximal value regardless of the sign (abs)
        output = 57.2958 * max(bend_series, key=abs)
    except AttributeError:
        output = np.nan
    return output


# Find Minimal Bend Amplitude of a given bout
def min_bend_amp(numBout, structure):
    """returns the minimal bend amplitude found in the bout number numBout,
    stored in structure raises NaN if you only have one bend in a bout"""
    series = pd.Series(structure[numBout]["Bend_Amplitude"])
    try:
        output = 57.2958 * (series.min())
    except AttributeError:
        output = np.nan
    return output


# Find amplitude of a given bout's first bend
def first_bend_amp(numBout, structure):
    """returns the amplitude of the first bend of the bout number numBout, stored in structure """
    try:
        structure[numBout]["Bend_Amplitude"][0]
    except IndexError:
        output = structure[numBout]["Bend_Amplitude"]
    else:
        if type(structure[numBout]["Bend_Amplitude"][0]) == int:
            output = np.nan
        else:
            output = structure[numBout]["Bend_Amplitude"][0] * 57.2958
    return output


# Find amplitude of a given bout's second bend
def second_bend_amp(numBout, structure):
    """returns the amplitude of the second bend of the bout number numBout, stored in structure """
    try:
        structure[numBout]["Bend_Amplitude"][1]
    except IndexError:
        output = np.nan
    else:
        output = structure[numBout]["Bend_Amplitude"][1] * 57.2958
    return output


# Find ratio of the amplitudes of first and second bend
def ratio_bend(numBout, structure):
    """returns the ratio between amplitude of the first bend and second bend
    of the bout number numBout, stored in structure """
    try:
        second_bend_amp(numBout, structure) / first_bend_amp(numBout, structure)
    except TypeError:
        output = np.nan
    else:
        output = second_bend_amp(numBout, structure) / first_bend_amp(numBout, structure)
    return output


# Find mean Tail Beat Frequency as the number of oscillations in a bout divided by bout duration
def mean_tbf(numBout, structure, fq):
    """returns the mean Tail Beat Frequency of the bout number numBout, stored both in dataset and in structure """
    return N_osc_b(numBout, structure) / bout_duration(numBout, structure, fq)


# return the list of the absolute frame indices corresponding to a bout's bends
def bend_indices(numBout, structure):
    """returns the absolute frame indices of the bout number numBout, stored in structure"""
    return list(structure[numBout]["Bend_TimingAbsolute"])


# Instant TBF for each bout
def bend_iTBF(bend_index, bend_indices, df_frame):
    bend_index_rel = find_indices(bend_indices, lambda e: e == bend_index)[0]
    output = np.nan
    if bend_index_rel > 0:
        previous_bend_index = bend_indices[bend_index_rel - 1]
        time_previous = df_frame.Time_index[previous_bend_index]
        time_actual = df_frame.Time_index[bend_index]
        output = 1 / (2 * (time_actual - time_previous))
    return output


def bout_iTBF(bout_num, structure, df_frame):
    bend_indices = pd.Series(structure[bout_num]["Bend_TimingAbsolute"])
    output = list(bend_indices.apply(bend_iTBF, args=(bend_indices, df_frame)))
    return output


def median_iTBF(bout_num, iTBF):
    try:
        output = pd.Series(iTBF[bout_num]).median()
    except ValueError:
        output = np.nan
    return output


def max_iTBF(bout_num, iTBF):
    try:
        output = pd.Series(iTBF[bout_num]).max()
    except ValueError:
        output = np.nan
    return output


# sum the tail angles at each frame composing a bout, after correction
def tail_angle_sum(bout_num, frame_dataset, supstruct):
    """returns the sum of all tail angle measured during bout bout_num"""
    bout_start = get_bout_start(bout_num, supstruct)
    bout_end = get_bout_end(bout_num, supstruct)
    return sum(frame_dataset.Tail_angle[bout_start:bout_end])


# Find mean Tail Angle of a given bout
def mean_tail_angle(bout_num, frame_dataset, supstruct):
    """
    Calculates the mea, of tail angle of each time points in a swim bout.

    :param bout_num: Bout number
    :return: mean value of the tail angle between start point and end point of the bout.
    """
    bout_start = get_bout_start(bout_num, supstruct)
    bout_end = get_bout_end(bout_num, supstruct)
    return frame_dataset.Tail_angle[bout_start:bout_end].mean()


def integral_ta(bout_num, frame_dataset, supstruct):
    """
    Calculates the integral of tail angle of the bout duration.

    :param bout_num: Bout number
    :return: integral of tail angle values at each time point between bout start and bout end.
    """
    bout_start = get_bout_start(bout_num, supstruct)
    bout_end = get_bout_end(bout_num, supstruct)
    return simps(frame_dataset.Tail_angle[bout_start:bout_end], dx=1)


# plot tail angle of all the bouts
def fig_tail_angle_all(bout_dataset, frame_dataset, n_bouts, n_col, n_row):
    """In one figure labeled 35, will plot in subplots the tail angle of the first n_bouts,
    of the bout_dataset, contained in the frame_dataset
    Plot will be divided in n_col columns, and n_rows columns"""
    plt.figure(35)
    plt.suptitle('Tail angle over time for each bout')
    plt.tight_layout()
    for i, index in enumerate(range(n_bouts)):
        plt.subplot(n_row, n_col, i + 1)
        color = 'b'
        start = bout_dataset.BoutStart_summed[index]
        end = bout_dataset.BoutEnd_summed[index]
        plt.plot(frame_dataset.Time_index[start:end], frame_dataset.Tail_angle[start:end], color)
        plt.plot(frame_dataset.Time_index[start:end], frame_dataset.Bend_Amplitude[start:end], 'rx', markersize=1.5)
        plt.ylim(-90, 90)
        plt.title(index)
        if i == ((n_row - 1) * n_col):
            plt.xlabel('time (in sec)')
            plt.ylabel('Tail angle')


def get_bend_amps(bout, df_frame):
    """
    Returns list of bends amplitude for a specific bout.

    :param bout:
    :param df_frame:
    :return:
    """

    return list(df_frame.loc[(df_frame.BoutNumber == bout) & (df_frame.Bend_Amplitude.notna()),
                             'Bend_Amplitude'])


def get_median_bend_amp(bout, df_frame):
    """
    Median bend amplitude of a bout.

    :param bout:
    :param df_frame:
    :return:
    """

    return np.nanmedian(get_bend_amps(bout, df_frame))


def build_ta_all_points(struct,supstruct, nb_tail_points):

    lastFrame = supstruct['lastFrame']
    ta_all = np.zeros((nb_tail_points-1, lastFrame))
    for i in range(len(struct)):
        start, end = struct[i]['BoutStart'], struct[i]['BoutEnd']
        for j in range(nb_tail_points-1):
            ta_all[j,start:end+1] = struct[i]['allTailAnglesSmoothed'][j]
    return ta_all


def build_ta_cumsum(struct, supstruct, nb_tail_points):

    ta_all = build_ta_all_points(struct, supstruct, nb_tail_points)

    return np.sum(ta_all, axis=0)
