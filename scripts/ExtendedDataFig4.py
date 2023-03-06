import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import shelve

plt.style.use('seaborn-poster')

data_path = '/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/ML_pipeline_output/fig6/'
df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/data_summary_BH.csv')


def resample(tail_angle, fps_bh, fps_ci):
    fq1 = str(round(1 / fps_bh, 6)) + 'S'
    fq2 = str(round(1 / fps_ci, 6)) + 'S'
    df_ta = pd.Series(np.abs(tail_angle), index=pd.date_range(start="00:00:00",
                                                              periods=len(tail_angle),
                                                              freq=fq1))
    ta_resampled = np.array(df_ta.resample(fq2).max())

    return ta_resampled


def filter_array(array, w):
    filtered_array = pd.Series(array).rolling(w, center=True).mean()
    filtered_array[filtered_array.isna()] = 0
    return np.array(filtered_array)


def compute_absolute_change(array, w):
    array_f = filter_array(array, w)
    output = [False] + [array_f[i] > array_f[i - 1] for i in range(1, len(array_f))]
    return np.multiply(output, 1)  # convert boolean into binary


# Explore distribution and correlation of motor parameters across experiments


dict_motor_feature = {}
dict_motor_feature_non_zero = {}
for exp in os.listdir(data_path):

    if os.path.isdir(data_path + exp):
        try:

            fishlabel = exp.split('_')[0] + '_' + exp.split('_')[1]
            plane = exp.split('_')[2] + '_' + exp.split('_')[3]
            print('\n\n', fishlabel, plane)

            output_path = df_summary.loc[
                (df_summary.fishlabel == fishlabel) & (df_summary.plane == plane), 'output_path'].item()

            shelve_out = shelve.open(output_path + '/shelve_calciumAnalysis.out')

            dff = shelve_out['dff_f_lp_inter']
            df_frame = pd.read_pickle(output_path + '/dataset/df_frame')
            tail_angle = shelve_out['tail_angle']
            fps_ci = shelve_out['fps']
            fps_bh = shelve_out['fps_beh']

            shelve_out.close()

            time_indices_2p = np.arange(dff.shape[1]) / fps_ci
            time_indices_bh = np.arange(len(tail_angle)) / fps_bh

        except (IndexError, FileNotFoundError):
            continue

        tail_angle_resampled = resample(tail_angle, fps_bh, fps_ci)


        def bin_mean_freq(frame):
            try:
                time = time_indices_2p[frame]
                next_time = time_indices_2p[frame + 1]
                bh_inf, bh_sup = int(time * fps_bh), int(next_time * fps_bh)
                n_osc = len(
                    df_frame[(df_frame.index.isin(range(bh_inf, bh_sup))) & (df_frame.Bend_Amplitude.notna())]) / 2
                if n_osc != 0:
                    output = n_osc * fps_ci
                else:
                    output = 0
            except IndexError:
                output = 0
            return output


        freq_array = pd.Series(np.arange(len(tail_angle_resampled))).apply(bin_mean_freq)

        max_frame_linReg = int(df_summary[(df_summary.fishlabel == fishlabel) &
                                          (df_summary.plane == plane)].max_frame_linReg.item())
        raw_arrays = dict()
        w = 5
        for i, variable in enumerate(['freq', 'amp']):

            if variable == 'freq':
                raw_array = freq_array[:max_frame_linReg]
            else:
                raw_array = tail_angle_resampled[:max_frame_linReg]

            for j, reg_type in enumerate(['abs', 'change']):

                if reg_type == 'abs':
                    raw_arrays[variable + '_' + reg_type] = raw_array
                elif reg_type == 'change':
                    raw_arrays[variable + '_' + reg_type] = compute_absolute_change(raw_array, w)

        non_zero_indices = np.where(raw_arrays['amp_abs'] != 0)[0]

        df_non_zero = pd.DataFrame({'fishlabel': fishlabel,
                                    'plane': plane,
                                    'exp': exp,
                                    'freq_abs': raw_arrays['freq_abs'][non_zero_indices],
                                    'freq_change': raw_arrays['freq_change'][non_zero_indices],
                                    'amp_abs': raw_arrays['amp_abs'][non_zero_indices],
                                    'amp_change': raw_arrays['amp_change'][non_zero_indices],
                                    })

        df = pd.DataFrame({'fishlabel': fishlabel,
                           'plane': plane,
                           'exp': exp,
                           'freq_abs': raw_arrays['freq_abs'],
                           'freq_change': raw_arrays['freq_change'],
                           'amp_abs': raw_arrays['amp_abs'],
                           'amp_change': raw_arrays['amp_change'],
                           })
        dict_motor_feature[exp] = df
        dict_motor_feature_non_zero[exp] = df_non_zero

df_all = pd.concat(dict_motor_feature, ignore_index=True)
df_all_non_zero = pd.concat(dict_motor_feature_non_zero, ignore_index=True)

print('\n\n number of frames in total:', len(df_all))
print('\n number of non zero frames in total', len(df_all_non_zero))

sns.displot(df_all_non_zero, x='freq_abs', kind='kde')
plt.ylim(0,0.13)
plt.savefig(data_path + '/Jan_2022/kde_iTBF.svg')
sns.displot(df_all_non_zero[df_all_non_zero.amp_abs < 30], x='amp_abs', kind='kde')
plt.ylim(0,0.13)
plt.savefig(data_path + '/Jan_2022/kde_iTA.svg')
plt.figure(figsize=(6,6))
sns.scatterplot(data=df_all_non_zero, x='freq_abs', y='amp_abs')
plt.ylim(0,30)
plt.tight_layout()
plt.savefig(data_path + '/Jan_2022/iTBFvsiTA.svg')
