import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
import seaborn as sns
import logging

from utils.behavior_vs_stimulation import get_onset_first_swim, behaving_prop_stims, behaving_prop_rests, \
    get_bout_period, behaving_prop_stims_per_cat, behaving_prop_rests_per_cat


plt.style.use('seaborn-poster')

master_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/Behavior/'
summary_csv = pd.read_csv(master_path + '/analysis_3/data_summary_BH_electrical_stim.csv')

handlers = [logging.FileHandler(master_path + '/analysis_3/logging_2.log'), logging.StreamHandler()]
logging.basicConfig(handlers=handlers,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)

fish = list(set(summary_csv.fishlabel))
clean_fish = [x for x in fish if str(x) != 'nan']

# TODO: get interbout duration

prop_all_stim = []
prop_all_rest = []
onset_first_stim = []
indices = []
fish_col = []
trial_col = []
stim_ints = []
stim_freqs = []
electrode_pos_all = []

dict_all = {}

prop_forward_all_stim = []
prop_forward_all_rest = []
prop_struggle_all_stim = []
prop_struggle_all_rest = []

for index in summary_csv.index:

    # if to be included
    if summary_csv.for_plot.iloc[index] == 1:
        # if summary_csv.include.iloc[index] == 1:

        # get experiment info
        fishlabel = summary_csv.fishlabel.iloc[index]
        ZZ_path = summary_csv.ZZ_path.iloc[index]

        trial = ZZ_path.split('/')[-1]
        if not trial or trial == 'ZZ_output':
            trial = ZZ_path.split('/')[-2]

        logging.info('\n\n' + fishlabel + '\n' + trial)

        indices.append(fishlabel + '_' + trial)
        fish_col.append(fishlabel)
        trial_col.append(trial)

        fps = int(summary_csv.frameRate.iloc[index])
        stim_intensity = float(summary_csv.stim_intensity.iloc[index])
        stim_freq = summary_csv.stim_freq.iloc[index]
        stim_dur = summary_csv.stim_dur.iloc[index].item()
        nStim = summary_csv.nStim.iloc[index].item()
        electrode_pos = summary_csv.electrode_pos.iloc[index]

        stim_freqs.append(stim_freq)
        stim_ints.append(stim_intensity)
        electrode_pos_all.append(electrode_pos)

        tail_angle = np.load(master_path + fishlabel + '/' + trial + '/dataset/tail_angle.npy')
        time_indices_bh = np.arange(len(tail_angle)) / fps

        abf = pyabf.ABF(summary_csv.stim_trace_path.iloc[index])

        try:
            df_bout = pd.read_pickle(master_path + fishlabel + '/' + trial + '/dataset/df_bout')
        except FileNotFoundError:
            logging.info('\nNo df _bout found for this one.')
            df_bout = pd.DataFrame()

        # Add additional info to dataframe

        df_bout['electrode_pos'] = electrode_pos

        # Get time at which behavior camera started

        channel_camera = [i for i, a in enumerate(abf.adcNames) if a in ['IN 0', 'IN 10', 'Behavior']][0]
        abf.setSweep(sweepNumber=0, channel=channel_camera)

        shift = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]

        #  Get proportion of time spent swimming during stim or not stim

        channel_stim = [i for i, a in enumerate(abf.adcNames) if a in ['Stim', 'Stim_OUT']][0]
        abf.setSweep(sweepNumber=0, channel=channel_stim)

        prop_on = behaving_prop_stims_per_cat(abf, time_indices_bh, tail_angle, shift, nStim, stim_dur)
        prop_bh_on, prop_f_on, prop_s_on = prop_on
        prop_all_stim.append(round(prop_bh_on, 2))
        prop_forward_all_stim.append(round(prop_f_on, 2))
        prop_struggle_all_stim.append(round(prop_s_on, 2))
        logging.info('\nProportion of time spent swimming during stim: ' + str(prop_bh_on))
        logging.info('\nProportion of time spent swimming forward during stim: ' + str(prop_f_on))
        logging.info('\nProportion of time spent struggling during stim: ' + str(prop_s_on))

        prop_rest = behaving_prop_rests_per_cat(abf, time_indices_bh, tail_angle, shift, nStim, stim_dur)
        prop_bh_rest, prop_f_rest, prop_s_rest = prop_rest
        prop_all_rest.append(round(prop_bh_rest, 2))
        prop_forward_all_rest.append(round(prop_f_rest,2))
        prop_struggle_all_rest.append(round(prop_s_rest,2))
        logging.info('\nProportion of time spent swimming during rest: ' + str(prop_bh_rest))
        logging.info('\nProportion of time spent swimming forward during rest: ' + str(prop_f_rest))
        logging.info('\nProportion of time spent struggling during rest: ' + str(prop_s_rest))


# TODO: create stim list
# TODO: add loop for each stim in trial !!!!!

df_2 = pd.DataFrame({'fishlabel': np.repeat(fish_col, 2),
                     'trial': np.repeat(trial_col, 2),
                     'prop_forward': [np.nan] * len(indices) * 2,
                     'prop_struggle': [np.nan] * len(indices) * 2,
                     'prop_swim': [np.nan] * len(indices) * 2,
                     'stim': np.repeat(stim_ints, 2),
                     'stim_int': np.repeat(stim_ints, 2),
                     'stim_freq': np.repeat(stim_freqs, 2),
                     'condition': np.tile(['rest', 'stim'], len(indices)),
                     'electrode_pos': np.repeat(electrode_pos_all, 2)})

df_2.loc[df_2.condition == 'rest', 'prop_forward'] = prop_forward_all_rest
df_2.loc[df_2.condition == 'stim', 'prop_forward'] = prop_forward_all_stim

df_2.loc[df_2.condition == 'rest', 'prop_swim'] = prop_all_rest
df_2.loc[df_2.condition == 'stim', 'prop_swim'] = prop_all_stim

df_2.loc[df_2.condition == 'rest', 'prop_struggle'] = prop_struggle_all_rest
df_2.loc[df_2.condition == 'stim', 'prop_struggle'] = prop_struggle_all_stim
