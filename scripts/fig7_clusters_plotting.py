import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shelve
import pyabf


save_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/V2a_recruitment_behavior/analysis_9/'
df_summary = pd.read_csv('/network/lustre/iss01/wyart/analyses/2pehaviour/MLR_analyses/data_summary_BH.csv')
plt.style.use('seaborn-poster')

df = pd.read_pickle(save_path + 'df.pkl')

exp = 0

output_path = df_summary.output_path[exp]
fps_bh = df_summary.frameRateBeh[exp]
fps_ci = df_summary.frameRate[exp]
tail_angle = np.load(output_path + '/dataset/tail_angle.npy')
df_bout = pd.read_pickle(output_path + '/dataset/df_bout')
df_syl = pd.read_pickle(output_path + '/dataset/df_syllabus_manual')
df_frame = pd.read_pickle(output_path + '/dataset/df_frame')
time_indices_bh = np.load(output_path + '/dataset/time_indices.npy')
fish = df_summary.fishlabel[exp]
plane = df_summary.plane[exp]
real_plane = df_summary.real_plane[exp]
data_path = df_summary.data_path[exp]
ops = np.load(data_path + '/suite2p/plane0/ops.npy', allow_pickle=True).item()
abf = pyabf.ABF(df_summary.stim_trace_path.iloc[exp])
# Get time at which behavior camera started
channel_camera = [i for i, a in enumerate(abf.adcNames) if a in ['IN 0', 'IN 10', 'Behavior']][0]
abf.setSweep(sweepNumber=0, channel=channel_camera)
shift = abf.sweepX[np.where(abf.sweepY > 1)[0][0]]
channel_stim = [i for i, a in enumerate(abf.adcNames) if a in ['Stim', 'Stim_OUT']][0]
abf.setSweep(sweepNumber=0, channel=channel_stim)
print('\n\n{}, {},\n{}'.format(fish, plane, output_path))


with shelve.open(output_path + '/shelve_calciumAnalysis.out') as f:
    cells = f['cells']
    dff = f['dff_c']
    dff_f = f['dff_f_avg']
    dff_f_lp = f['dff_f_lp']
    noise = f['noise']
    noise_f = f['noise_f_avg']
    stat = f['stat']


cluster_1 = list(df[(df.fishlabel == fish) & (df.plane == plane) &
                    (df.final_cell_group == 'medulla') & (df.added != 1)].cell.unique())
unwanted_cells = [50,24,0,2]
cluster_mmed = [i for i in cluster_1 if i not in unwanted_cells]
cluster_lmed = list(df[(df.fishlabel == fish) & (df.plane == plane) &
                       (df.final_cell_group == 'medullar_lateral') & (df.added != 1)].cell.unique())


# Build Dataframe of signal


def build_df_signal(roi, dff_f, dff_f_lp, time_indices, cluster):
    df = pd.DataFrame(columns=['cell', 'time_point', 'dff_f', 'dff_f_lp', 'cluster'],
                      index=np.arange(dff_f.shape[1]))
    df['cell'] = roi
    df['time_point'] = time_indices
    df['dff_f_avg'] = dff_f[roi]
    df['dff_f_lp'] = dff_f_lp[roi]
    df['cluster'] = cluster
    return df


time_indices_ci = np.arange(dff_f.shape[1])/fps_ci
dict_df_signal = {}
for i, cell in enumerate(cluster_mmed+cluster_lmed):
    cluster = 'latMed' if cell in cluster_lmed else 'mMed'
    dict_df_signal[cell] = build_df_signal(int(cell), dff_f, dff_f_lp, time_indices_ci, cluster)
df_signal = pd.concat(dict_df_signal, ignore_index=True)

pal = {'mMed': "#32313f", 'latMed': "#7ba375"}
plt.figure()
plt.title('Low pass filtered signal')
sns.lineplot(data=df_signal, x='time_point', y='dff_f_lp', hue='cluster', palette=pal)
plt.figure()
plt.title('Average filtered signal')
sns.lineplot(data=df_signal, x='time_point', y='dff_f', hue='cluster', palette=pal)

plt.figure()
plt.plot(time_indices_ci, np.mean(dff_f[[int(i) for i in cluster_mmed], :], axis=0), color='darkorange')
plt.plot(time_indices_ci, np.mean(dff_f[[int(i) for i in cluster_lmed], :], axis=0), color='limegreen')
plt.savefig(save_path + '/mean_avgf_trace_exp0.svg')



