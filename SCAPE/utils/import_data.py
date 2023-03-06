import numpy as np
import pandas as pd
import os
import pickle
import logging
import scipy.io


def load_summary_csv(url):
    df = pd.read_csv(url.replace('/edit#gid=', '/export?format=csv&gid='))
    return df


class ReAnalyze(Exception):
    pass


def load_experiment(path, fishlabel):
    """Loads the experiment object corresponding the the Exp class. This object contains all the info about the
    experiment performed on the fish adressed by fishlabel."""
    with open(path + 'exps/' + fishlabel + '_exp', 'rb') as handle:
        experiment = pickle.load(handle)
    return experiment


def load_behavior_dataframe(exp):

    df_frame = pd.read_pickle(exp.savePath + exp.runID + '/df_frame.pkl')
    df_bout = pd.read_pickle(exp.savePath + exp.runID + '/df_bout.pkl')

    return df_frame, df_bout
    

def load_suite2p_outputs(fishlabel, trial, input_path):
    """Load every output that the suite2p gives you
    Arguments given are fishlabel, real_trial_num and folder_path.
    If folder_path is not given, automatically check for the data path in the summary csv file.
    You can change the path to the summary csv file here in the function.
    If folder_path is give,;
    Returns F, Fneu, spks, stat, ops, iscell"""
    if not os.path.exists(input_path):
        raise FileNotFoundError('Path to your folder is not valid.')
    try:
            F = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/F.npy', allow_pickle=True)
            Fneu = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/Fneu.npy', allow_pickle=True)
            spks = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/spks.npy', allow_pickle=True)
            stat = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/stat.npy', allow_pickle=True)
            ops = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/ops.npy', allow_pickle=True)
            ops = ops.item()
            iscell = np.load(input_path + fishlabel + '/' + trial + '/suite2p/plane0' + '/iscell.npy', allow_pickle=True)
    except FileNotFoundError:
        F = np.load(input_path + fishlabel + '/' + trial + '/F.npy', allow_pickle=True)
        Fneu = np.load(input_path + fishlabel + '/' + trial + '/Fneu.npy', allow_pickle=True)
        spks = np.load(input_path + fishlabel + '/' + trial + '/spks.npy', allow_pickle=True)
        stat = np.load(input_path + fishlabel + '/' + trial + '/stat.npy', allow_pickle=True)
        ops = np.load(input_path + fishlabel + '/' + trial + '/ops.npy', allow_pickle=True)
        ops = ops.item()
        iscell = np.load(input_path + fishlabel + '/' + trial + '/iscell.npy', allow_pickle=True)
    return F, Fneu, spks, stat, ops, iscell


def load_bout_object(path, fishlabel, trial):
    with open(path + 'dataset/' + fishlabel + '/' + trial + '/bouts', 'rb') as handle:
        bouts = pickle.load(handle)
    return bouts


def load_mat_file(path):
    mat = scipy.io.loadmat(path)['info'][0]
    mdtype = mat['info'].dtype
    ndata = {n: mat['info'][n][0, 0] for n in mdtype.names}
    mdtype_daq = ndata['daq'].dtype
    daq_data = {n: ndata['daq'][n][0, 0] for n in mdtype_daq.names}
    return ndata, daq_data


def load_tail_angle(exp):
    try:
        return np.load(exp.savePath + exp.runID +'/tail_angle.npy')
    except FileNotFoundError:
        print('Tail angle numpy trace was not found at this location:\n', exp.savePath + exp.runID +'/tail_angle.npy')
        return None


def load_summary_csv(url):
    df = pd.read_csv(url.replace('/edit#gid=', '/export?format=csv&gid='))
    return df
