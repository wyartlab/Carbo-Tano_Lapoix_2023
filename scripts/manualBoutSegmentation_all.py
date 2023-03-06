import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.integrate import simps
import traceback
from pylab import arange, plot, sin, ginput, show
import os


#  Manual segmentation of bouts into syllabus - eg. motor components which can be exhibited in the same discrete event


def get_interbout_interval(bout, df, fps):
    """

    Computes, for a given bout, the invert of the time between beginning of this bout and end of the previous bout.

    :param bout: int, bout number
    :param df: dataframe, with bouts parameters
    :param fps: frame rate of behavior camera
    :return: float, invert of the time between bout and previous bout

    """
    try:
        previous_end = df.end[bout - 1]
        output = (df.start[bout] - previous_end) / fps
    except KeyError:
        output = np.nan

    return output


def get_cat(bout, df, th):
    """

    Automatic categorization of the bout using simple thresholding on the maximal bend amplitude (absolute).

    :param bout: int, bout number
    :param df: dataframe, with bouts parameters
    :param th: float, threshold of max tail angle below which a bout is considered a forward.
    :return: str, F for forward and S for struggle.

    """
    if df.abs_max_bend_amp[bout] < th:
        output = 'F'
    else:
        output = 'S'
    return output


def get_cat_bends_ratio(bout, df, th):
    """

    Automatic categorization of the bout using proportion of bends below or above a threshold.
    If there are more than 5 bends, and more than 60% of the bends are below a given threshold, categorize as forward.
    Otherwise, categorize as struggle.

    :param bout: int, bout number
    :param df: dataframe, with bouts parameters
    :param th: float, threshold of max tail angle below which a bout is considered a forward.
    :return: str, F for forward and S for struggle.

    """
    bend_amps = np.abs(np.array(df.bend_amps[bout]))
    if (len(np.where(bend_amps < th)[0]) >= 0.6 * len(bend_amps)) & (len(bend_amps) > 5):
        output = 'F'
    else:
        output = 'S'
    return output


def get_real_i_tbf(df_frame, syllabus_mask):
    """

    Computes, for each cycle, the instantaneous tail beat frequency as the invert of the time between each cycle.
    Loops through bends and for each even bend, the time spent between this bend and the previous even bend.

    :param df_frame: dataframe, with information on tail angle and bend at each frame.
    :param syllabus_mask: boolean mask of the dataframe filtering for when a syllabus/episode was happening.
    :return: list of floats, corresponding to the iTBF for each cycle.

    """
    even_bend = False  # first bend will be uneven
    iTBF = []
    j = 0  # initialise first cycle
    for bend in df_frame[syllabus_mask & (df_frame.Bend_Amplitude.notna())].index:  # for all bend during syllabus
        if j == 0:  # if first bend of the first cycle, put NaN
            iTBF.append(np.nan)
            previous_bend = bend  #  but store this bend index as the index of beginning of the first cycle
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
df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses'
                         '/data_summary_BH_electrical_stim.csv')
dict_all_exp = {}

for exp in df_summary.index:

    try:
        # Load corresponding behavior
        try:
            split_path = df_summary.df_behavior_path[exp].split('/')[0:-2]
        except AttributeError:
            continue
        output_path = '/'.join(split_path)
        fps = df_summary.frameRate[exp]

        tail_angle = np.load(output_path + '/dataset/tail_angle.npy')
        df_bout = pd.read_pickle(output_path + '/dataset/df_bout')
        df_frame = pd.read_pickle(output_path + '/dataset/df_frame')
        date = df_summary.date[exp]
        fish = df_summary.fish[exp]
        fishlabel = df_summary.fishlabel[exp]
        print('\n\n{}, {},\n{}'.format(date, fish, output_path))

        if df_bout.empty:
            continue

    except FileNotFoundError:
        traceback.print_exc()
        continue

    #  this was temporary not to re-segment something that already worked.
    if (exp in dict_all_exp.keys()) or (os.path.exists(output_path + '/dataset/df_syllabus_manual_bends_ratio')):
        if fishlabel != '210225_F01':
            print('\nalready segmented.\n')
            continue

    # While loop asking user to click on beginning and end of bout

    dict_all = {}

    stop_seg = False
    i = 0
    previous_end = 0
    while not stop_seg:

        # First plot with full data to select the window to zoom in
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Tail angle [°]')
        ax.plot(df_frame.Tail_angle, color='silver')
        plt.plot([previous_end], [df_frame.Tail_angle[previous_end]], 'x')
        plt.title(
            'fish: {}, fishlabel {}, syllabus {}:\n1 - click on the two corners of the area to enlarge'.format(fish,
                                                                                                               fishlabel,
                                                                                                               i),
            fontsize=12)
        try:
            zoom_inf, zoom_sup = plt.ginput(2)
        except ValueError:
            stop_seg = True

        x_inf, _ = zoom_inf
        if x_inf < 0:
            x_inf = 0
        x_sup, _ = zoom_sup
        if x_sup > len(df_frame.Tail_angle):
            x_sup = len(df_frame.Tail_angle)

        plt.close('all')

        #  Second plot with the zoomed window to select precisely beginning and end of bout
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Tail angle [°]')
        plt.plot(df_frame.Tail_angle[int(x_inf):int(x_sup)], 'silver')
        if previous_end >= x_inf:
            plt.plot([previous_end], [df_frame.Tail_angle[previous_end]], 'x')
        plt.title('Syllabus {}:\nselect start and end of syllabus'.format(i), fontsize=12)

        ginput = plt.ginput(2)
        try:
            a, b = ginput
            frame_start, _ = a
            frame_end, _ = b
            start = int(frame_start)
            end = int(frame_end)

            # check that there is no overlapping with previous bout, or that the bout doesn't end after the end of trace
            if start < previous_end:
                start = previous_end + 1
            if end >= len(df_frame):
                end = len(df_frame) - 1

            syllabus_mask = df_frame.index.isin(np.arange(start, end + 1))

            #  Ask user for manual category
            plt.close('all')
            plt.figure(figsize=(4, 2))
            plt.text(1, 0.5, 'S', fontsize=17, color='white', fontfamily='sans-serif',
                     bbox=dict(boxstyle='round', facecolor='magenta', alpha=0.5))
            plt.text(0.5, 0.5, 'F', fontsize=17, color='white', fontfamily='sans-serif',
                     bbox=dict(boxstyle='round', facecolor='royalblue', alpha=0.5))
            plt.xlim(0.2, 1.2)
            plt.ylim(0.2, 0.8)
            plt.title('Choose manual cat')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            x, y = plt.ginput(1)[0]
            if x <= 0.7:
                manual_cat = 'F'
            else:
                manual_cat = 'S'

            #  Build dataframe with bout info

            dict_syllabus = pd.DataFrame({'start': start,
                                          'end': end,
                                          'duration': round(df_frame.Time_index[end] - df_frame.Time_index[start], 3),
                                          'n_oscillations': len(
                                              df_frame[(df_frame.index.isin(np.arange(start, end + 1))) &
                                                       (df_frame.Bend_Amplitude.notna())]),
                                          'max_bend_amp': max(df_frame.Tail_angle[start:end + 1], key=abs),
                                          'min_bend_amp': min(df_frame.Tail_angle[start:end + 1], key=abs),
                                          'abs_max_bend_amp': abs(max(df_frame.Tail_angle[start:end + 1], key=abs)),
                                          'integral_ta': simps(df_frame.Tail_angle[start:end + 1]),
                                          'sum_ta': np.sum(df_frame.Tail_angle[start:end + 1]),
                                          'iTBF': [np.nan],
                                          'bend_amps': [np.nan],
                                          'median_bend_amp': np.nan,
                                          'manual_cat': manual_cat
                                          }, dtype=object)

            iTBF = get_real_i_tbf(df_frame, syllabus_mask)
            dict_syllabus.at[0, 'iTBF'] = iTBF
            dict_syllabus.at[0, 'bend_amps'] = list(df_frame.loc[syllabus_mask & (df_frame.Bend_Amplitude.notna()),
                                                                 'Bend_Amplitude'])
            dict_syllabus['median_bend_amp'] = np.nanmedian(list(dict_syllabus['bend_amps']))
            dict_syllabus['mean_TBF'] = float(dict_syllabus['n_oscillations'] / dict_syllabus['duration'])
            dict_syllabus['median_iTBF'] = np.nanmedian(iTBF)
            dict_syllabus['mean_iTBF'] = np.nanmean(iTBF)

            ax.plot([start, end], [df_frame.Tail_angle[start], df_frame.Tail_angle[end]], 'x', color='orange')
            dict_all[i] = dict_syllabus
            previous_end = end
            i += 1

        except ValueError:
            traceback.print_exc()
            stop_seg = True

    #  end of segmentation

    df_syllabus = pd.concat(dict_all, ignore_index=True)

    # Category & IBI
    df_syllabus['IBI'] = pd.Series(df_syllabus.index).apply(get_interbout_interval, args=(df_syllabus, fps))
    df_syllabus['Cat'] = pd.Series(df_syllabus.index).apply(get_cat, args=(df_syllabus, 25))
    df_syllabus['Cat_bends_ratio'] = pd.Series(df_syllabus.index).apply(get_cat_bends_ratio, args=(df_syllabus, 25))

    # Plot individual syllabus
    dict_colors = {'F': 'darkorange',
                   'S': 'darkblue',
                   'M': 'mediumturquoise'}

    plt.style.use('ggplot')
    n_row = math.ceil(len(df_syllabus) / 3)
    fig, ax = plt.subplots(n_row, 3, figsize=(12, 12), sharey=True)
    fig.suptitle('\n{}, manual cat color (F:orange, S:blue)')

    for bout in df_syllabus.index:
        ax_plot = ax.flatten()[bout]

        if bout == 0:
            ax_plot.set_ylabel('Tail angle [°]')

        start, end = df_syllabus.start[bout], df_syllabus.end[bout]
        cat = df_syllabus.manual_cat[bout]
        bends = np.array(df_frame['Bend_Amplitude'].loc[start:end])
        ax_plot.plot(df_frame.Time_index[start:end + 1], tail_angle[start:end + 1], color=dict_colors[cat])
        ax_plot.plot(df_frame.Time_index[start:end + 1], bends, 'o')
        ax_plot.grid(False)
    ax_plot.set_xlabel('Time [s]')

    plt.savefig(output_path + '/fig/tail_angle_all_syllabus_bends_ratio.svg')
    plt.close()

    #  Save df

    continue_loop = input('Continue ?')
    if continue_loop == 'n':
        break

    dict_all_exp[exp] = df_syllabus
    df_syllabus.to_pickle(output_path + '/dataset/df_syllabus_manual_bends_ratio')
