"""

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
import seaborn as sns
import logging
import ast


def prop_swim_stim_cat(abf, time_indices_bh, shift, nStim, stim_dur, df_bouts, tail_angle, fps):
    """
    For a given experiment, returns proportion of time spent swimming, and time spent swimming forward or struggling,
    for each given stimulation.
    1) find, for each stim, start and end time of the given stimulation from the abf trace
    2) find, in behavior referential, corresponding frames, and extract bouts which STARTED during the stimulation
    3) calculate nominator of the fraction : for each of these bouts, count number of frames, and classify if it was forward or struggle
    4) calculate denominator of the fraction : either between stim start or stim end, but if a bout started during stim and continued after,
    total number of frames is from stim start to last frame of the last bout.
    5) calculate proportion as numerator / denominator

    :param abf:
    :param time_indices_bh:
    :param shift:
    :param nStim:
    :param stim_dur:
    :param df_bouts:
    :param tail_angle:
    :param fps:
    :return:
    """
    prop_on, prop_f, prop_s = ([], [], [])  # initialise

    for stim in np.arange(nStim):  # for each stimulation

        num_on, num_f, num_s = (0, 0, 0)  # initialise numerator to calculate proportion

        # 1) find stim start and end

        if stim == 0:  #   if first stim
            stim_start = abf.sweepX[
                np.where(abf.sweepY > 1)[0][0]]  #   stim start is first index at which stim trace went up

        else:  # for other stims,
            end_index = int(np.where(abf.sweepX == stim_end)[
                                0])  # calc stim start at first index at which stim trace went up after end index of
            # previous stim
            stim_start = abf.sweepX[end_index:][np.where(abf.sweepY[end_index:] > 1)[0][0]]

        stim_end = stim_start + stim_dur

        #  2) find corresponding frames in behavioral traces and bouts which STARTED during the stim

        # get corresponding frame in behavior signal
        frame_start = np.argmin([abs(i - stim_start) for i in time_indices_bh + shift])
        frame_end = np.argmin([abs(i - stim_end) for i in time_indices_bh + shift])
        indices = np.arange(frame_start, frame_end + 1)  # behavior frames during which stimulation happened

        bouts_stim = df_bouts[df_bouts.start.isin(indices)].index  # get bouts that started during this stimulation

        # 3) Calculate nominator: number of frames where fish moved during stim

        bout_end = np.nan  # if no bout happened, make sure bout_end is not empty
        for bout in bouts_stim:  # for each of these bouts
            bout_start, bout_end = df_bouts.start.iloc[bout], df_bouts.end.iloc[bout]

            # check if bout ended after stim !
            if bout_end > frame_end:
                bout_end = frame_end

            num_on += bout_end - bout_start  # number of frames during which it happened

            #  count time spent swimming forward or struggle by looking at max ta during bins of behavior
            if df_bouts.manual_cat.iloc[bout]:
                num_f += bout_end - bout_start
            else:
                num_s += bout_end - bout_start

        # 4) Calculate denominator: total number of frames during stim

        den = frame_end - frame_start  # denominator is total number of frames between

        #  5) Calculate proportion

        # stim start and last frame taken into account (either last frame of last bout or end of stimulation)
        # calculate the proportion
        prop_on.append(round(num_on / den, 4) * 100)
        prop_f.append(round(num_f / den, 4) * 100)
        prop_s.append(round(num_s / den, 4) * 100)

    return prop_on, prop_f, prop_s


def prop_swim_rest_cat(abf, time_indices_bh, shift, nStim, stim_dur, df_bouts, tail_angle, fps):
    """
    For a given experiment, returns proportion of time spent swimming, and time spent swimming forward or struggling,
    for each given period of rest between stimulation..

    1) find, for each stim, start and end time of the given stimulation from the abf trace
    2) find, in behavior referential, corresponding frames, and extract bouts which started before the stimulation.
    3) calculate nominator of the fraction : for each of these bouts, count number of frames, and classify if it was forward or struggle
    4) calculate denominator of the fraction : either between stim start or stim end, but if a bout started during stim and continued after,
    total number of frames is from stim start to last frame of the last bout.
    5) calculate proportion as numerator / denominator

    :param abf:
    :param time_indices_bh:
    :param shift:
    :param nStim:
    :param stim_dur:
    :param df_bouts:
    :param tail_angle:
    :param fps:
    :return:
    """

    prop_on, prop_f, prop_s = ([], [], [])  # initialise
    previous_end = 0  #  starts the resting period at the beginning of the recording

    for stim in np.arange(nStim):  # for each stimulation
        num_on, num_f, num_s = (0, 0, 0)  # initialise numerator to calculate proportion

        # 1) find stim start and end

        if stim == 0:
            stim_start = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]

        else:
            end_index = int(np.where(abf.sweepX == stim_end)[0])
            stim_start = abf.sweepX[end_index:][np.where(abf.sweepY[end_index:] > 1)[0][0]]

        stim_end = stim_start + stim_dur

        #  2) find corresponding frames in behavioral traces and bouts which started before stim

        # get corresponding frame in behavior signal
        frame_start = np.argmin([abs(i - stim_start) for i in time_indices_bh + shift])
        frame_end = np.argmin([abs(i - stim_end) for i in time_indices_bh + shift])
        indices = np.arange(previous_end, frame_start)  # behavior frames between previous stim and stim start
        bouts_rest = df_bouts[
            df_bouts.start.isin(indices)].index  # get bouts that started during this resting period

        for bout in bouts_rest:  # for each of these bouts

            bout_start, bout_end = df_bouts.start.iloc[bout], df_bouts.end.iloc[bout]

            # check if no bout ended after the resting period !
            if bout_end > frame_start:
                bout_end = frame_start

            if df_bouts.manual_cat.iloc[bout] == 'F':
                num_f += bout_end - bout_start
            else:
                num_s += bout_end - bout_start

        den = frame_start - previous_end  # total number of frames for this chunk of resting time

        prop_on.append(round(num_on / den, 4) * 100)
        prop_f.append(round(num_f / den, 4) * 100)
        prop_s.append(round(num_s / den, 4) * 100)

        previous_end = frame_end  # end of the stim becomes the previous end for the next sresting period

    return prop_on, prop_f, prop_s


def get_first_bout_type(abf, time_indices_bh, shift, nStim, stim_dur, df_bouts, fps, tail_angle):
    """
    For a given stimulation, compute the behavior type of the first bout which happened during stimulation.
    1) Find corresponding behavior frames during which stimulation happened.
    2) Find if any bout happened during these frames.
        2-1) If so, find the bout type of the first behavior
        2-2) If bout was mixed, look at first 100ms of the bout, attribute bout type given the angles during this period
    3) If no bout happened, output NaN.

    :param abf:
    :param time_indices_bh:
    :param shift:
    :param nStim:
    :param stim_dur:
    :param df_bouts:
    :param fps:
    :param tail_angle:
    :return: list of size nStim, each element is a string corresponding to first bout type of the given stim.
    """
    output = []
    for stim in np.arange(nStim):  # for each stimulation

        if stim == 0:
            stim_start = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]

        else:
            end_index = int(np.where(abf.sweepX == stim_end)[0])
            stim_start = abf.sweepX[end_index:][np.where(abf.sweepY[end_index:] > 1)[0][0]]

        stim_end = stim_start + stim_dur

        # get corresponding frame in behavior signal
        frame_start = np.argmin([abs(i - stim_start) for i in time_indices_bh + shift])
        frame_end = np.argmin([abs(i - stim_end) for i in time_indices_bh + shift])
        indices = np.arange(frame_start, frame_end)  # behavior frames during which stimulation happened

        try:

            first_bout = df_bouts[df_bouts.start.isin(indices)].index[0]
            first_bout_type = df_bouts.manual_cat.iloc[first_bout]

        except IndexError:  # if no behavior during this stim
            first_bout_type = np.nan

        output.append(first_bout_type)

    return output


def get_onset_first_swim(nStim, stim_dur, abf, time_indices_bh, df_bout, shift, fps):
    """
    For a given stimulation, calculates the time between beginning of the stimulation and the first bout triggered by
    the stimulation, if there is any.
    1) Find behavior frames corresponding to stimulation.
    2) Find if any behavior started in these frames.
    3) If there is, calculate time between onset of the first bout and the stim start.
    4) If none, output NaN.

    :param nStim:
    :param stim_dur:
    :param abf:
    :param time_indices_bh:
    :param df_bout:
    :param shift:
    :param fps:
    :return:

    """
    output = []

    for stim in np.arange(nStim):

        if stim == 0:
            stim_start = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]

        else:
            end_index = int(np.where(abf.sweepX == stim_end)[0])
            stim_start = abf.sweepX[end_index:][np.where(abf.sweepY[end_index:] > 1)[0][0]]

        stim_end = stim_start + stim_dur

        # get corresponding frame in behavior signal
        frame_start = np.argmin([abs(i - stim_start) for i in time_indices_bh + shift])
        frame_end = np.argmin([abs(i - stim_end) for i in time_indices_bh + shift])
        indices = np.arange(frame_start, frame_end)  # behavior frames during which stimulation happened

        try:

            first_start = list(df_bout.loc[df_bout.start.isin(indices), 'start'])[0]

            output.append(round((first_start - frame_start) / fps, 3))

        except (IndexError, KeyError):
            output.append(np.nan)

    return output


def get_bout_stim_num(bout, df_bout, time_indices_bh, abf, shift, nStim, stim_dur):
    start = df_bout.start.iloc[bout]
    time_start = time_indices_bh[start] + shift
    output = np.nan

    for stim in range(nStim):
        if stim == 0:  #   if first stim
            stim_start = abf.sweepX[
                np.where(abf.sweepY > 1)[0][0]]  #   stim start is first index at which stim trace went up

        else:  # for other stims,
            end_index = int(np.where(abf.sweepX == stim_end)[
                                0])  # calc stim start at first index at which stim trace went up after end index of
            # previous stim
            stim_start = abf.sweepX[end_index:][np.where(abf.sweepY[end_index:] > 1)[0][0]]

        stim_end = stim_start + stim_dur

        if stim_start <= time_start <= stim_end:
            output = stim

    return output


def get_bout_period(bout, df_bout, time_indices_bh, abf, shift, nStim, stim_dur):
    stim_num = get_bout_stim_num(bout, df_bout, time_indices_bh, abf, shift, nStim, stim_dur)

    if np.isnan(stim_num):
        output = 'during_rest'
    else:
        output = 'during_stim'

    return output


def get_bout_stim_freq(bout, df_bout, freqs, time_indices_bh, abf, shift, nStim, stim_dur):
    if len(freqs) == 1:
        output = freqs[0]
    else:
        stim_num = get_bout_stim_num(bout, df_bout, time_indices_bh, abf, shift, nStim, stim_dur)
        if not np.isnan(stim_num):
            output = freqs[stim_num]
        else:
            output = np.nan

    return output


def get_interbout_interval(bout, df_bout, fps):
    try:
        previous_end = df_bout.end[bout - 1]
        output = (df_bout.start[bout] - previous_end) / fps
    except KeyError:
        output = np.nan

    return output


master_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/Behavior/'
summary_csv = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/'
                          'data_summary_BH_electrical_stim.csv')

handlers = [logging.FileHandler(master_path + '/analysis_10/logging.log'), logging.StreamHandler()]
logging.basicConfig(handlers=handlers,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)

fish = list(set(summary_csv.fishlabel))
clean_fish = [x for x in fish if str(x) != 'nan']

prop_on_stim_all, prop_f_stim_all, prop_s_stim_all = ([], [], [])
prop_on_rest_all, prop_f_rest_all, prop_s_rest_all = ([], [], [])
# prop_f_bend_all = []
# prop_s_bend_all = []
n_bouts_all = []
n_struggle_all = []
n_forward_all = []
onset_first_stim = []
first_bout_type_all = []
indices = []
fish_col = []
trial_col = []
electrode_placement_col = []
stim_ints = []
stim_freqs = []
stim_durs = []
electrode_pos_all = []

dict_all = {}
dict_bends_all = {}

# _ = pd.Series(summary_csv.index).apply(run_ZZ_extraction, args=(summary_csv, master_path))
# plt.close()

plt.style.use('seaborn-poster')
# pd.Series(clean_fish).apply(plot_ta_vs_stim, args=(summary_csv, master_path))

for index in summary_csv.index:

    if summary_csv.include[index] == 1:

        # get experiment info
        fishlabel = summary_csv.fishlabel[index]
        ZZ_path = summary_csv.ZZ_path[index]

        trial = ZZ_path.split('/')[-1]
        if not trial or trial == 'ZZ_output':
            trial = ZZ_path.split('/')[-2]

        logging.info('\n\n' + fishlabel + '\n' + trial)
        nStim = summary_csv.nStim[index]

        try:
            df_bout = pd.read_pickle(master_path + fishlabel + '/' + trial + '/dataset/df_syllabus_manual_bends_ratio')
            df_frame = pd.read_pickle(master_path + fishlabel + '/' + trial + '/dataset/df_frame')
            df_bout['fishlabel'] = fishlabel
            df_bout['trial'] = trial
        except FileNotFoundError:
            logging.info('\nNo df _bout found for this one.')
            continue

        indices.extend([fishlabel + '_' + trial] * nStim)
        fish_col.extend([fishlabel] * nStim)
        trial_col.extend([trial] * nStim)

        electrode_placement = summary_csv.electrode_placement[index]
        electrode_placement_col.extend([electrode_placement] * nStim)

        fps = int(summary_csv.frameRate[index])
        stim_intensity = summary_csv.stim_intensity.iloc[index]
        stim_freq = [float(i) for i in summary_csv.stim_freq[index].split(',')]  #  same exp can have stim at different
        # freq
        if len(stim_freq) == 1:
            stim_freq = stim_freq * nStim

        stim_dur = summary_csv.stim_dur[index]
        electrode_pos = summary_csv.electrode_pos[index]

        stim_freqs.extend(stim_freq)
        stim_ints.extend([stim_intensity] * nStim)
        stim_durs.extend([stim_dur] * nStim)

        electrode_pos_all.extend([electrode_pos] * nStim)

        tail_angle = np.load(master_path + fishlabel + '/' + trial + '/dataset/tail_angle.npy')
        time_indices_bh = np.arange(len(tail_angle)) / fps

        abf = pyabf.ABF(summary_csv.stim_trace_path.iloc[index])



        # Get time at which behavior camera started

        channel_camera = [i for i, a in enumerate(abf.adcNames) if a in ['IN 0', 'IN 10', 'Behavior']][0]
        abf.setSweep(sweepNumber=0, channel=channel_camera)

        shift = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]

        #  Get proportion of time spent swimming during stim or not stim

        channel_stim = [i for i, a in enumerate(abf.adcNames) if a in ['Stim', 'Stim_OUT']][0]
        abf.setSweep(sweepNumber=0, channel=channel_stim)

        prop_bh_on, prop_f_on, prop_s_on = prop_swim_stim_cat(abf, time_indices_bh, shift, nStim, stim_dur, df_bout,
                                                              tail_angle, fps)
        prop_on_stim_all.extend(prop_bh_on)
        prop_f_stim_all.extend(prop_f_on)
        prop_s_stim_all.extend(prop_s_on)
        logging.info('\nProportion of time spent swimming during stim: ' + str(prop_bh_on) +
                     '\n(forward: {}%, struggle: {}'.format(prop_f_on, prop_s_on))

        prop_bh_rest, prop_f_rest, prop_s_rest = prop_swim_rest_cat(abf, time_indices_bh, shift, nStim, stim_dur,
                                                                    df_bout, tail_angle, fps)
        prop_on_rest_all.extend(prop_bh_rest)
        prop_f_rest_all.extend(prop_f_rest)
        prop_s_rest_all.extend(prop_s_rest)
        logging.info('\nProportion of time spent swimming during rest: ' + str(prop_bh_rest) +
                     '\n(forward: {}%, struggle: {}'.format(prop_f_rest, prop_s_rest))

        # get first bout cat

        first_bout_type = get_first_bout_type(abf, time_indices_bh, shift, nStim, stim_dur, df_bout, fps, tail_angle)

        first_bout_type_all.extend(first_bout_type)
        logging.info('\nFirst bout type: {}'.format(first_bout_type))

        #  Get onset to first swim

        onset = get_onset_first_swim(nStim, stim_dur, abf, time_indices_bh, df_bout, shift, fps)

        onset_first_stim.extend(onset)
        logging.info('\nTime onset to first bout: {}'.format(onset))

        # Add additional info to dataframe

        df_bout['electrode_pos'] = electrode_pos
        df_bout['electrode_placement'] = electrode_placement
        df_bout['stim_duration'] = stim_dur
        df_bout['stim_intensity'] = stim_intensity
        df_bout['IBI'] = pd.Series(df_bout.index).apply(get_interbout_interval, args=(df_bout, fps))
        df_bout['stim_num'] = pd.Series(df_bout.index).apply(get_bout_stim_num,
                                                             args=(
                                                             df_bout, time_indices_bh, abf, shift, nStim, stim_dur))
        df_bout['stim_freq'] = pd.Series(df_bout.index).apply(get_bout_stim_freq,
                                                              args=(df_bout, stim_freq, time_indices_bh, abf,
                                                                    shift, nStim, stim_dur))

        #  Get if bouts happened during stim or during rest
        df_bout['bout_condition'] = pd.Series(df_bout.index).apply(get_bout_period,
                                                                   args=(df_bout, time_indices_bh, abf,
                                                                         shift, nStim, stim_dur))

        # create df bends
        # df_bend = create_df_bend(df_bout, df_frame)
        # dict_bends_all[fishlabel + trial] = df_bend
        #
        # # compute mean and median power for each bout
        #
        # df_bout['mean_power'] = [np.nanmean(df_bend.loc[df_bend.BoutNum == bout, 'instant_power']) for bout in
        #                          df_bout.index]
        # df_bout['median_power'] = [np.nanmedian(df_bend.loc[df_bend.BoutNum == bout, 'instant_power']) for bout in
        #                            df_bout.index]

        # Get new category of bout
        # df_bout['new_cat'] = pd.Series(df_bout.index).apply(get_bout_new_category, args=(df_bend,))

        # # Get for each stim, nBouts and nBouts of each type:
        n_bouts_all.extend([len(df_bout[df_bout.stim_num == i]) for i in range(nStim)])
        n_struggle_all.extend([len(df_bout[(df_bout.stim_num == i) & (df_bout.manual_cat == 'S')]) for i in range(nStim)])
        n_forward_all.extend([len(df_bout[(df_bout.stim_num == i) & (df_bout.manual_cat == 'F')]) for i in range(nStim)])

        dict_all[fishlabel + trial] = df_bout

df_bout_all = pd.concat(dict_all, axis=0, join='outer', sort=False, ignore_index=True)
# df_bend_all = pd.concat(dict_bends_all, axis=0, join='outer', sort=False, ignore_index=True)

df = pd.DataFrame({'fishlabel': np.repeat(fish_col, 2),
                   'trial': np.repeat(trial_col, 2),
                   'electrode_placement': np.repeat(electrode_placement_col, 2),
                   'prop': [np.nan] * len(indices) * 2,
                   'stim': np.repeat(stim_ints, 2),
                   'stim_int': np.repeat(stim_ints, 2),
                   'stim_freq': np.repeat(stim_freqs, 2),
                   'stim_duration': np.repeat(stim_durs, 2),
                   'condition': np.tile(['rest', 'stim'], len(indices)),
                   'electrode_pos': np.repeat(electrode_pos_all, 2),
                   'onset_first_swim': np.repeat(onset_first_stim, 2),
                   'first_bout_type': np.repeat(first_bout_type_all, 2)})

df.loc[df.condition == 'rest', 'prop'] = prop_on_rest_all
df.loc[df.condition == 'stim', 'prop'] = prop_on_stim_all

df.loc[df.condition == 'rest', 'prop_s'] = prop_s_rest_all
df.loc[df.condition == 'stim', 'prop_s'] = prop_s_stim_all

df.loc[df.condition == 'stim', 'prop_f'] = prop_f_stim_all
df.loc[df.condition == 'rest', 'prop_f'] = prop_f_rest_all


df.loc[df.condition == 'stim', 'n_bouts'] = n_bouts_all
df.loc[df.condition == 'stim', 'n_bouts_s'] = n_struggle_all
df.loc[df.condition == 'stim', 'n_bouts_f'] = n_forward_all
df.loc[df.condition == 'stim', 'prop_s_bouts'] = df.loc[df.condition == 'stim', 'n_bouts_s'] / df.loc[df.condition == 'stim', 'n_bouts']
df.loc[df.condition == 'stim', 'prop_f_bouts'] = df.loc[df.condition == 'stim', 'n_bouts_f'] / df.loc[df.condition == 'stim', 'n_bouts']

for i in df.query("condition == 'stim'").index:
    if df.n_bouts[i] == 0:
        output = np.nan
    else:
        if df.n_bouts_f[i] == 0:
            df.at[i, 'ratio_f_s'] = -1
        elif df.n_bouts_s[i] == 0:
            df.at[i, 'ratio_f_s'] = 1
        else:
            df.at[i, 'ratio_f_s'] = (df.n_bouts_f[i] - df.n_bouts_s[i]) / df.n_bouts[i]
    print(i)


# TODO: locate where this dataframe was saved, it's probably df_electrode
df_electorde_placement = pd.DataFrame(
    columns=['date', 'fishlabel', 'electrode_placement', 'ratio_f_s', 'median_ratio_f_s'],
    index=range(len(set(df.electrode_placement))))
for i in set(df.electrode_placement):
    date = list(df.query("electrode_placement == @i & condition == 'stim'").fishlabel)[0].split('_')[0]
    df_electorde_placement.at[i, 'date'] = date
    df_electorde_placement.at[i, 'fishlabel'] = \
    list(df.query("electrode_placement == @i & condition == 'stim'").fishlabel)[0]
    df_electorde_placement.at[i, 'electrode_pos'] = \
    list(df.query("electrode_placement == @i & condition == 'stim'").electrode_pos)[0]
    df_electorde_placement.at[i, 'electrode_placement'] = i
    ratio_f_s = list(df.query("electrode_placement == @i & condition == 'stim'").ratio_f_s)
    df_electorde_placement.at[i, 'ratio_f_s'] = ratio_f_s
    df_electorde_placement.at[i, 'median_ratio_f_s'] = np.nanmedian(ratio_f_s)
    
    
df_bout_all.to_csv(master_path + '/analysis_10/df_bout_all.csv')
df.to_csv(master_path + '/analysis_10/df.csv')

df_bout_all.to_pickle(master_path + '/analysis_10/df_bout_all')
df.to_pickle(master_path + '/analysis_10/df')
