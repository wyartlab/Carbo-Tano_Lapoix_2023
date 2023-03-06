import pandas as pd
import numpy as np
from utils.functions_ZZ_extraction import load_ZZ_output


def load_clustering_results(path):
    output = pd.read_pickle(path)
    return output['dfParam'], output['pca_result']


def reassign_bout_cluster(id, df_bout_all, df_clustering):
    start, end = df_bout_all.BoutStart[id], df_bout_all.BoutEnd[id]
    numBout = df_bout_all.NumBout[id]
    try:
        id_c = df_clustering[(df_clustering.BoutStart == start) & (df_clustering.BoutEnd == end) &
                             (df_clustering.NumBout == numBout)].index[0]

        classification = df_clustering.classification[id_c]
    except (SyntaxError, IndexError):
        classification = np.nan
    return classification


def build_df_bout_trace_cluster(df_bout_all, nTimeSteps=70):

    nBouts = len(df_bout_all)
    time_steps = np.arange(nTimeSteps)/300
    output = pd.DataFrame({'bout_index': np.repeat(df_bout_all.index, nTimeSteps),
                           'tail_angle': np.nan,
                           'time_point': np.tile(time_steps, nBouts),
                           'frame': np.tile(range(nTimeSteps), nBouts),
                           'condition': np.repeat(df_bout_all.enucleated, nTimeSteps),
                           'stage': np.repeat(df_bout_all.stage, nTimeSteps),
                           'classification': np.repeat(df_bout_all.classification, nTimeSteps),
                           'fishlabel': np.repeat(df_bout_all.Fishlabel, nTimeSteps)})

    # Prior to looping into all bouts, load all struct so we don't have to reload them
    dict_struct = dict()
    for exp in df_bout_all.exp.unique():
        dict_struct[exp] = load_exp_struct(exp, df_bout_all)
    print('Loaded all struct')

    for index in df_bout_all.index:
        exp = df_bout_all.exp[index]
        struct = dict_struct[exp]
        boutNum = df_bout_all.NumBout[index]

        #  load tail angle for this bout
        bout_trace = get_bout_trace(boutNum, struct)

        #  check that length is more than 70 frames, or correct it
        if len(bout_trace) < nTimeSteps:
            bout_trace_temp = np.zeros((nTimeSteps))
            bout_trace_temp[:] = np.nan
            bout_trace_temp[0:len(bout_trace)] = bout_trace
            bout_trace = bout_trace_temp

        output.loc[output.bout_index == index, 'tail_angle'] = bout_trace[0:nTimeSteps]

        if 'manual_cat' in df_bout_all.columns:
            output['manual_cat'] = np.repeat(df_bout_all.manual_cat, nTimeSteps)
        #  get bout classification
        # output.loc[output.bout_index == index, 'classification'] = df_bout_all.classification[index]
        if index % 100 == 0:
            print('Proccessed {}/{} bouts'.format(index, len(df_bout_all)))

    return output


def load_exp_struct(exp, df_bout_all):
    path = df_bout_all.loc[df_bout_all.exp == exp, 'path'].unique()[0]
    struct = load_ZZ_output(path)
    return struct['wellPoissMouv'][0][0]


def get_bout_trace(boutNum, struct):
    return np.array(struct[boutNum]['TailAngle_smoothed'])*57.2958
