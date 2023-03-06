import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils_behavior_dataset import *
import utils.functions_ZZ_extraction as fct


df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/summaryData.csv')
index = 18

fishlabel = df_summary.fishlabel[index]
run = df_summary.run[index]
ZZ_path = df_summary.ZZ_path[index] + '/'
output_path = df_summary.output_path[index] + '/'
nbtailpoints = 20
fps_ci = df_summary.frameRateSCAPE[index]
fps_beh = df_summary.frameRateBeh[index]
nFramesBeh = int(df_summary.nFrames_beh[index])
nFramesSCAPE = int(df_summary.nFrames_SCAPE[index])
if not os.path.exists(output_path):
    os.mkdir(output_path)

print('\n fish: {}\nexp: {}'.format(fishlabel, run))

pd.options.mode.chained_assignment = None

#  Load the ZZ output txt file
filepath = fct.load_zz_results(ZZ_path)
with open(filepath) as f:
    big_supstruct = json.load(f)
    supstruct = big_supstruct["wellPoissMouv"][0][0]


# Creating a DataFrame contaning information on bout, fish position... at each frame

# Defining index of the DataFrame as the frame number
# number of bouts in file
NBout = len(supstruct)
print('\nNumber of bouts:' + str(NBout))

# Building Tail Angle trace

time_indices = np.arange(nFramesBeh)/fps_beh
tail_angle = np.array([np.nan] * nFramesBeh)
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

        for point in range(nbtailpoints - 1):

            try:
                tail_angles[point, bout_start:bout_end] = 57.2958 * np.array(
                    supstruct[bout]["allTailAnglesSmoothed"][point])
            except ValueError:
                try:
                    tail_angles[point, bout_start - 1:bout_end] = 57.2958 * np.array(
                        supstruct[bout]["allTailAnglesSmoothed"][point])
                except ValueError:
                    pass

    if not supstruct[bout]['Bend_Amplitude']: # if bout does not have any bend (false bout detection)
        continue
    else:
        bouts.append(bout)
        struct.append(supstruct[bout])

print('New n bouts:' + str(len(bouts)))

tail_angle = np.nan_to_num(tail_angle)

tail_angle_sum = tail_angles.sum(axis=0)

np.save(output_path + '/dataset/tail_angle.npy', np.array(tail_angle))
np.save(output_path + '/dataset/tail_angles.npy', np.array(tail_angles))
# np.save(output_path + '/dataset/tail_angle_sum.npy', np.array(tail_angle_sum))
np.save(output_path + '/dataset/time_indices.npy', time_indices)

#  Buidling DataFrame with tail angle

# create index of dataframe: step is one frame
# range(x) gives you x values, from 0 to x-1. So here you have to add +1
index_frame = pd.Series(range(nFramesBeh))
# Creating empty DataFrame
df_frame = pd.DataFrame({'fishlabel': fishlabel,
                         'run': run,
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
iTBF = df_bout_index.apply(get_real_iTBF, args=(struct, fps_beh))
median_iTBF = df_bout_index.apply(fct.median_iTBF, args=(pd.Series(list(iTBF), index=df_bout_index),))
mean_tail_angle = df_bout_index.apply(fct.mean_tail_angle, args=(df_frame, struct))
ta_sum = df_bout_index.apply(fct.tail_angle_sum, args=(df_frame, struct))
integral_tail_angle = df_bout_index.apply(fct.integral_ta, args=(df_frame, struct))
bend_amp_all = df_bout_index.apply(fct.get_bend_amps, args=(df_frame,))
median_bend_amp_all = df_bout_index.apply(fct.get_median_bend_amp, args=(df_frame,))

df_bout = pd.DataFrame({'fishlabel': fishlabel,
                        'run': run,
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
                        'Side_biais': pd.Series(np.nan, index=df_bout_index),
                        'bends_amp': bend_amp_all,
                        'median_bend_amp': median_bend_amp_all
                        }, index=df_bout_index)

print(df_bout.describe())

df_bout.to_pickle(output_path + '/dataset/df_bout.pkl')
df_frame.to_pickle(output_path + '/dataset/df_frame.pkl')

print(run + 'done. \n\n')
# Build tail angle reformated to fit suite2P GUI

fq1 = str(round(1/fps_beh, 6))+'S'
fq2 = str(round(1/fps_ci, 6))+'S'
tail_angle_s2p_gui(tail_angle, int(nFramesSCAPE), fq1, fq2, output_path)
