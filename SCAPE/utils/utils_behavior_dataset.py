import json
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import traceback

import utils.functions_ZZ_extraction as fct
from utils.import_data import *


class NoOutputError(Exception):
    pass


def bout_type(bout, df_bout):
    if df_bout['abs_Max_Bend_Amp'].iloc[bout] < 25:
        cat = 'F'  #  forward only

    elif (df_bout['abs_Max_Bend_Amp'].iloc[bout] > 25) & (df_bout['Bout_Duration'].iloc[bout] < 0.6):
        cat = 'S'  # struggle only

    else:
        cat = 'M'  # mix of both

    return cat


def get_real_iTBF(bout, struct, fq):
    # TODO: build second function to get half beat iTBF
    even_bend = False
    iTBF = []
    for bend, time_index in enumerate(struct[bout]["Bend_TimingAbsolute"]):
        if bend == 0:
            iTBF.append(np.nan)
            previous_time = time_index
            continue
        elif not even_bend:
            iTBF.append(np.nan)
            even_bend = True
        else:
            iTBF.append(round(1 / (time_index - previous_time)*fq, 3))
            even_bend = False
            previous_time = time_index
    return iTBF


def get_iHBF(bout, struct, fq):
    # TODO: build second function to get half beat iTBF
    iHBF = []
    for bend, time_index in enumerate(struct[bout]["Bend_TimingAbsolute"]):
        if bend == 0:
            iHBF.append(np.nan)
            previous_time = time_index
            continue
        else:
            iHBF.append(round(1 / (time_index - previous_time)*fq, 3))
            previous_time = time_index
    return iHBF


def extract_behavior_fish(i, fishlabel, ZZ_path, df_summary, nbtailpoints=20):

    trial = df_summary.run[i]

    fps = df_summary.frameRateBeh[i]

    # nFrames_beh = df_summary.loc[
    #     (df_summary.fishlabel == fishlabel) & (df_summary.ZZ_path == ZZ_path), 'nFrames_beh'].item()

    print('\n fish: {}\nexp: {}'.format(fishlabel, trial))

    pd.options.mode.chained_assignment = None

    ZZ_path = ZZ_path + '/'

    #  Load the ZZ output txt file
    # If no txt file, go on to the next loop indent
    filepath = fct.load_zz_results(ZZ_path)

    # Open txt file as a struct
    if filepath:
        with open(filepath) as f:
            big_supstruct = json.load(f)
            supstruct = big_supstruct["wellPoissMouv"][0][0]
    else:
        raise NoOutputError

    # Creating a DataFrame contaning information on bout, fish position... at each frame

    # Defining index of the DataFrame as the frame number
    # number of bouts in file
    NBout = len(supstruct)
    print('\nNumber of bouts:' + str(NBout))

    # Building Tail Angle trace

    # if np.isnan(nFrames_beh): # if nFrames in behavior recording not filled
    #     nFrames_beh = supstruct[-1]['BoutEnd']+1

    nFrames_beh = supstruct[-1]['BoutEnd'] + 1
    nFrames_beh = int(nFrames_beh)
    tail_angle = np.array([np.nan] * nFrames_beh)
    tail_angles = np.zeros((nbtailpoints, len(tail_angle)))
    bouts = []

    struct = []

    for bout in range(NBout):

        bout_start = int(supstruct[bout]["BoutStart"])
        bout_end = int(supstruct[bout]["BoutEnd"])

        try:
            tail_angle[bout_start - 1:bout_end] = 57.2958 * np.array(supstruct[bout]["TailAngle_smoothed"])
        except ValueError:  #  if no tail angle registered for this bout
            pass

        if "allTailAnglesSmoothed" in supstruct[bout]:  # if all tail angle were calculated

            for point in range(nbtailpoints-1):

                try:
                    tail_angles[point, bout_start:bout_end+1] = 57.2958 * np.array(
                            supstruct[bout]["allTailAnglesSmoothed"][point])
                except ValueError:
                    pass

        if not supstruct[bout]['Bend_Amplitude']:
            continue
        else:
            bouts.append(bout)
            struct.append(supstruct[bout])

    print('New n bouts:' + str(len(bouts)))
    del supstruct

    tail_angle = np.nan_to_num(tail_angle)

    tail_angle_sum = tail_angles.sum(axis=0)
    time_indices = np.arange(nFrames_beh) / fps

    # np.save(output_path + '/dataset/tail_angle.npy', np.array(tail_angle))
    # np.save(output_path + '/dataset/tail_angles.npy', np.array(tail_angles))
    # np.save(output_path + '/dataset/tail_angle_sum.npy', np.array(tail_angle_sum))
    # np.save(output_path + '/dataset/time_indices.npy', time_indices)

    #  Buidling DataFrame with tail angle

    # create index of dataframe: step is one frame
    # range(x) gives you x values, from 0 to x-1. So here you have to add +1
    index_frame = pd.Series(range(nFrames_beh))
    # Creating empty DataFrame
    df_frame = pd.DataFrame({'Fishlabel': fishlabel,
                             'Trial': trial,
                             'Time_index': time_indices,
                             'BoutNumber': np.nan,
                             'Tail_angle': tail_angle,
                             'Bend_Index': np.nan,
                             'Instant_TBF': np.nan,
                             'Instant_HBF': np.nan,
                             'Bend_Amplitude': np.nan}, index=index_frame)

    # Filling this DataFrame

    # Creating a DataFrame containing start frame index and end frame index

    # Filling first the frames with a bout
    # using functions to find bout number and number of oscillations of this bout correspond to the frame
    # boutstart summed corresponds to the start frame of a given bout if all the videos were just one big video
    df_frame.BoutNumber = pd.Series(df_frame.index).apply(fct.bout_num, args=(struct, len(bouts)))
    df_frame.Bend_Index = pd.Series(df_frame.index).apply(fct.bend_index, args=(struct, len(bouts)))
    df_frame.Bend_Amplitude = pd.Series(df_frame.index).apply(fct.bend_amplitude, args=(struct, len(bouts)))

    # Creating another DataFrame containing quite the same info but per bout
    # index is the number of the bouts
    df_bout_index = pd.Series(np.arange(len(bouts)))
    num_bouts = bouts
    num_osc = df_bout_index.apply(fct.N_osc_b, args=(struct,))
    bout_duration = df_bout_index.apply(fct.bout_duration, args=(struct, fps))
    bout_start = df_bout_index.apply(fct.get_bout_start, args=(struct,))
    bout_end = df_bout_index.apply(fct.get_bout_end, args=(struct,))
    max_bend_amp = df_bout_index.apply(fct.max_bend_amp, args=(struct,))
    min_bend_amp = df_bout_index.apply(fct.min_bend_amp, args=(struct,))
    first_bend_amp = df_bout_index.apply(fct.first_bend_amp, args=(struct,))
    second_bend_amp = df_bout_index.apply(fct.second_bend_amp, args=(struct,))
    ratio_first_second_bend = df_bout_index.apply(fct.ratio_bend, args=(struct,))
    mean_TBF = df_bout_index.apply(fct.mean_tbf, args=(struct, fps))
    iTBF = df_bout_index.apply(get_real_iTBF, args=(struct, fps))
    median_iTBF = df_bout_index.apply(fct.median_iTBF, args=(pd.Series(list(iTBF), index=df_bout_index),))
    max_iTBF = df_bout_index.apply(fct.max_iTBF, args=(pd.Series(list(iTBF), index=df_bout_index),))
    iHBF = df_bout_index.apply(get_iHBF, args=(struct, fps))
    median_iHBF = df_bout_index.apply(fct.median_iTBF, args=(pd.Series(list(iHBF), index=df_bout_index),))
    mean_tail_angle = df_bout_index.apply(fct.mean_tail_angle, args=(df_frame, struct))
    ta_sum = df_bout_index.apply(fct.tail_angle_sum, args=(df_frame, struct))
    integral_tail_angle = df_bout_index.apply(fct.integral_ta, args=(df_frame, struct))
    bend_amp_all = df_bout_index.apply(fct.get_bend_amps, args=(df_frame,))
    median_bend_amp_all = df_bout_index.apply(fct.get_median_bend_amp, args=(df_frame,))

    flags = [struct[i].get('flag') for i in df_bout_index]

    df_bout = pd.DataFrame({'Fishlabel': fishlabel,
                            'Trial': trial,
                            'path': ZZ_path,
                            'NumBout': num_bouts,
                            'Bout_Duration': bout_duration,
                            'BoutStart': bout_start,
                            'BoutEnd': bout_end,
                            'Number_Osc': num_osc,
                            'Max_Bend_Amp': max_bend_amp,
                            'abs_Max_Bend_Amp': abs(max_bend_amp),
                            'Min_Bend_Amp': min_bend_amp,
                            'First_Bend_Amp': first_bend_amp,
                            'Second_Bend_Amp': second_bend_amp,
                            'Ratio First Second Bend': ratio_first_second_bend,
                            'mean_TBF': mean_TBF,
                            'iTBF': iTBF,
                            'median_iTBF': median_iTBF,
                            'max_iTBF': max_iTBF,
                            'iHBF': iHBF,
                            'median_iHBF': median_iHBF,
                            'mean_tail_angle': mean_tail_angle,
                            'Tail_angle_sum': ta_sum,
                            'Integral_TA': integral_tail_angle,
                            'Side_biais': pd.Series(np.nan, index=df_bout_index),
                            'bends_amp': bend_amp_all,
                            'median_bend_amp': median_bend_amp_all,
                            'flag': flags
                            }, index=df_bout_index)

    print(df_bout.describe())

    if isinstance(df_summary.savePath[i], str):
        savePath = df_summary.savePath[i] + '/' + trial
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        df_frame.to_pickle(savePath + '/df_frame')
        df_bout.to_pickle(savePath + '/df_bout')
        print('DataFrames saved to pickle to:\n' + savePath)
        np.save(savePath + '/tail_angle.npy', np.array(tail_angle))
        np.save(savePath + '/tail_angles.npy', np.array(tail_angles))
        np.save(savePath + '/tail_angles_sum.npy', np.array(tail_angle_sum))
        print('Tail angles saved to numpy to:\n' + savePath)
    else:
        print('/!\ No savePath for exp', i)

    # if len(df_bout) > 1:
    #     dict_colors = {'F': 'darkorange',
    #                    'S': 'darkblue',
    #                    'M': 'mediumturquoise'}
    #
    #     plt.style.use('ggplot')
    #     n_row = math.ceil(len(bouts) / 3)
    #     fig, ax = plt.subplots(n_row, 3, figsize=(12, 12), sharey=True)
    #
    #     fig.suptitle(fishlabel + '\ntrial')
    #
    #     for bout in df_bout_index:
    #         ax_plot = ax.flatten()[bout]
    #
    #         if bout == 0:
    #             ax_plot.set_ylabel('Tail angle [°]')
    #
    #         start, end = df_bout['BoutStart'].iloc[bout], df_bout['BoutEnd'].iloc[bout]
    #         # cat = df_bout['Cat'].iloc[bout]
    #         cat = 'M'
    #         bends = np.array(df_frame['Bend_Amplitude'].loc[start:end])
    #         ax_plot.plot(time_indices[start:end + 1], tail_angle[start:end + 1], color=dict_colors[cat])
    #         ax_plot.plot(time_indices[start:end + 1], bends, 'o')
    #         ax_plot.grid(False)
    #         ax_plot.set_title('Bout ' + str(bout))
    #     ax_plot.set_xlabel('Time [s]')
    #
    #     plt.savefig(output_path + 'fig/tail_angle_all.svg')

    print(trial+ 'done. \n\n')

    return df_bout, df_frame


def run_ZZ_extraction(index, summary_csv):
    df_bout = {}
    df_frame = {}
    if summary_csv.includeBehavior[index] == 1:

        fishlabel = summary_csv.fishlabel[index]
        trial = summary_csv.run[index]
        ZZ_path = summary_csv.ZZ_path[index]
        if os.path.exists(ZZ_path):

            print('\n\n', fishlabel, '\n' + trial)

            try:
                df_bout, df_frame = extract_behavior_fish(index, fishlabel, ZZ_path, summary_csv, nbtailpoints=20)
                print('Number of bouts: ', len(df_bout))
            except (KeyError, IndexError, NoOutputError):
                traceback.print_exc()

        else:
            traceback.print_exc()
            print('No ZZ path for ' + fishlabel + '_' + str(index))

    return df_bout, df_frame


def tail_angle_s2p_gui(tail_angle, nFrames_2p, fq1, fq2, path):
    """

    From tail angle sampled at behavior camera rate, build two arrays to be loaded in the suite2P GUI,
    both of length nFrames in the suite2P file:

    1 array of tail angle resampled by summing the tail angle in embedded time points
    1 array binary for when there was a behavior in each time point.

    :tail_angle: numpy array, time series of tail angle value over time.
    :nFrames_2p: nb of time points (frames) in the calcium imaging data.
    :fq1: time step for behavior data, format 'X.XXXS'
    :fq2: time step for calcium imaging data, format 'X.XXXS'
    :path: path in which to save the output arrays.

    """

    df_ta = pd.Series(tail_angle, index=pd.date_range(start="00:00:00",
                                                      periods=len(tail_angle),
                                                      freq=fq1))
    ta_resampled = np.array(df_ta.resample(fq2).sum())
    ta_resampled_median = np.array(df_ta.resample(fq2).median())

    if len(ta_resampled) < nFrames_2p:
        ta_resampled = np.concatenate((ta_resampled, np.zeros(nFrames_2p-len(ta_resampled))))
        ta_resampled_median = np.concatenate((ta_resampled_median, np.zeros(nFrames_2p-len(ta_resampled_median))))
    elif len(ta_resampled) > nFrames_2p:
        ta_resampled = ta_resampled[:nFrames_2p]
        ta_resampled_median = ta_resampled_median[:nFrames_2p]

        # binary format
    ta_binary = np.zeros(ta_resampled.shape)
    mov_indices, = np.nonzero(ta_resampled)
    ta_binary[mov_indices] = 1

    np.save(path + 'tail_angle_resampled.npy', ta_resampled)
    np.save(path + 'tail_angle_binary.npy', ta_binary)
    np.save(path + 'tail_angle_resampled_median.npy', ta_resampled_median)

    return ta_resampled, ta_resampled_median, ta_binary
