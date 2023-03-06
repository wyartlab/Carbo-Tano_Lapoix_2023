import json
import pandas as pd
import math
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../utils/')
import functions_ZZ_extraction as fct
from import_data import *


# Initialisation

summary_file_path = sys.argv[1]
fishlabel = str(sys.argv[2])
trial = str(sys.argv[3])

df_summary = pd.read_csv(summary_file_path)
nbTailPoints = 0

output_path = df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == trial), 'output_path'].item()
raw_data_path = df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == trial), 'ZZ_path'].item()
fps_beh = df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == trial), 'frameRateBeh'].item()
nFrames_beh = df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == trial), 'nFrames_beh'].item()

script = os.path.basename(__file__)
handlers = [logging.FileHandler(output_path + '/logs/' + script + '.log'), logging.StreamHandler()]
logging.basicConfig(handlers=handlers,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Start

logging.info('\n fish: {}\nexp: {}'.format(fishlabel, trial))

pd.options.mode.chained_assignment = None

# Necessity given the struct of the output
numWell = 0
numBout = 0

#  Load the ZZ output txt file
# If no txt file, go on to the next loop indent
filepath = fct.load_zz_results(raw_data_path)

# Open txt file as a struct
with open(filepath) as f:
    big_supstruct = json.load(f)
    supstruct = big_supstruct["wellPoissMouv"][numWell][0]

# Creating a DataFrame contaning information on bout, fish position... at each frame

# Defining index of the DataFrame as the frame number
# number of bouts in file
NBout = len(supstruct)
logging.info('\nNumber of bouts:'+str(NBout))

# Building Tail Angle trace

tail_angle = np.array([np.nan] * int(nFrames_beh))
tail_angles = np.zeros((nbTailPoints, len(tail_angle)))
bouts = []

struct = []

for bout in range(NBout):
    bout_start = int(supstruct[bout]["BoutStart"])
    bout_end = int(supstruct[bout]["BoutEnd"])

    try:
        tail_angle[bout_start - 1:bout_end] = 57.2958 * np.array(supstruct[bout]["TailAngle_smoothed"])
    except ValueError: # if no tail angle registered for this bout
        pass

    if "allTailAnglesSmoothed" in supstruct[bout]: # if all tail angle were calculated

        for point in range(nbTailPoints - 1):

            try:
                tail_angles[point, bout_start:bout_end] = 57.2958 * np.array(
                    supstruct[bout]["allTailAnglesSmoothed"][point])
            except ValueError:
                tail_angles[point, bout_start - 1:bout_end] = 57.2958 * np.array(
                    supstruct[bout]["allTailAnglesSmoothed"][point])

    if not supstruct[bout]['Bend_Amplitude']:
        continue
    else:
        bouts.append(bout)
        struct.append(supstruct[bout])

logging.info('New n bouts:'+str(len(bouts)))

tail_angle = np.nan_to_num(tail_angle)  # fill with 0 when no tail movment detected

tail_angle_sum = tail_angles.sum(axis=0)
time_indices = np.arange(nFrames_beh) / fps_beh

np.save(output_path + '/dataset/tail_angle', np.array(tail_angle))
np.save(output_path + '/dataset/tail_angles', np.array(tail_angle))
np.save(output_path + '/dataset/tail_angle_sum', np.array(tail_angle))
np.save(output_path + '/dataset/time_indices', time_indices)

# Build tail angle reformated to fit suite2P GUI

nFrames_2p = int(df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == trial), 'nFrames_2p'])
fps_ci = float(df_summary.loc[(df_summary.fishlabel == fishlabel) & (df_summary.plane == trial), 'frameRate'])
fq1 = str(round(1/fps_beh, 6))+'S'
fq2 = str(round(1/fps_ci, 6))+'S'
fct.tail_angle_s2p_gui(tail_angle, nFrames_2p, fq1, fq2, output_path + '/dataset/')

#  Buidling DataFrame with tail angle

# create index of dataframe: step is one frame
# range(x) gives you x values, from 0 to x-1. So here you have to add +1
index_frame = pd.Series(np.arange(nFrames_beh))
# Creating empty DataFrame
df_frame = pd.DataFrame({'Name': fishlabel + '_' + trial,
                         'Time_index': time_indices,
                         'BoutNumber': np.nan,
                         'Tail_angle': tail_angle,
                         'Bend_Index': np.nan,
                         'Instant_TBF': np.nan,
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
num_osc = df_bout_index.apply(fct.N_osc_b, args=(struct,))
bout_duration = df_bout_index.apply(fct.bout_duration, args=(struct, fps_beh))
bout_start = df_bout_index.apply(fct.get_bout_start, args=(struct,))
bout_end = df_bout_index.apply(fct.get_bout_end, args=(struct,))
max_bend_amp = df_bout_index.apply(fct.max_bend_amp, args=(struct,))
min_bend_amp = df_bout_index.apply(fct.min_bend_amp, args=(struct,))
first_bend_amp = df_bout_index.apply(fct.first_bend_amp, args=(struct,))
second_bend_amp = df_bout_index.apply(fct.second_bend_amp, args=(struct,))
ratio_first_second_bend = df_bout_index.apply(fct.ratio_bend, args=(struct,))
mean_TBF = df_bout_index.apply(fct.mean_tbf, args=(struct, fps_beh))
iTBF = df_bout_index.apply(fct.bout_iTBF, args=(struct, df_frame))
median_iTBF = df_bout_index.apply(fct.median_iTBF, args=(pd.Series(list(iTBF), index=df_bout_index),))
mean_tail_angle = df_bout_index.apply(fct.mean_tail_angle, args=(df_frame, struct))
ta_sum = df_bout_index.apply(fct.tail_angle_sum, args=(df_frame, struct))
integral_tail_angle = df_bout_index.apply(fct.integral_ta, args=(df_frame, struct))

df_bout = pd.DataFrame({'Name': pd.Series(fishlabel + '_' + trial, index=df_bout_index),
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
                        'mean_tail_angle': mean_tail_angle,
                        'Tail_angle_sum': ta_sum,
                        'Integral_TA': integral_tail_angle,
                        'Side_biais': pd.Series(np.nan, index=df_bout_index)
                        }, index=df_bout_index)

logging.info(df_bout.describe())

# bout type


def bout_type(bout, df_bout):
    if df_bout['abs_Max_Bend_Amp'].iloc[bout] < 25:
        cat = 'F'  #  forward only

    elif (df_bout['abs_Max_Bend_Amp'].iloc[bout] > 25) & (df_bout['Bout_Duration'].iloc[bout] < 0.6):
        cat = 'S'  # struggle only

    else:
        cat = 'M'  # mix of both

    return cat


df_bout['Cat'] = df_bout_index.apply(bout_type, args=(df_bout, ))
logging.info('Cateogry of bouts: \n{} forward bouts,\n{} struggle,\n{} mixed'.format(len(df_bout[df_bout.Cat == 'F']),
                                                                              len(df_bout[df_bout.Cat == 'S']),
                                                                              len(df_bout[df_bout.Cat == 'M'])))


df_frame.to_pickle(output_path + 'dataset/df_frame')
df_bout.to_pickle(output_path + 'dataset/df_bout')
logging.info('DataFrames saved to pickle to:\n'+output_path+'dataset/')

dict_colors = {'F': 'darkorange',
               'S': 'darkblue',
               'M': 'mediumturquoise'}

plt.style.use('ggplot')
n_row = math.ceil(len(bouts)/3)
fig, ax = plt.subplots(n_row, 3, figsize=(12, 12), sharey=True)

for bout in df_bout_index:
    ax_plot = ax.flatten()[bout]

    if bout == 0:
        ax_plot.set_ylabel('Tail angle [°]')

    start, end = df_bout['BoutStart'].iloc[bout], df_bout['BoutEnd'].iloc[bout]
    cat = df_bout['Cat'].iloc[bout]
    bends = np.array(df_frame['Bend_Amplitude'].loc[start:end])
    ax_plot.plot(time_indices[start:end + 1], tail_angle[start:end + 1], color=dict_colors[cat])
    ax_plot.plot(time_indices[start:end + 1], bends, 'o')
    ax_plot.grid(False)
ax_plot.set_xlabel('Time [s]')

plt.savefig(output_path + 'fig/tail_angle_all.svg')

logging.info(trial+' done. \n\n')
