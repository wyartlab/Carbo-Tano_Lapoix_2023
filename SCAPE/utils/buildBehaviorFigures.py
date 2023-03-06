import numpy as np
import pandas as pd
import json
from utils import functions_ZZ_extraction as loadzz


def load_df_bout_all(path):
    return pd.read_pickle(path)


def load_ta(path):
    return np.load(path+'/tail_angle.npy')


def load_zz_output(path):
    with open(loadzz.load_zz_results(path)) as f:
        supstruct = json.load(f)
        lastFrame = supstruct['lastFrame']
        struct = supstruct["wellPoissMouv"][0][0]
    return lastFrame, struct


def build_cum_ta_single_bout(bout, struct, ta):
    return ta


def build_cum_ta(struct, lastFrame):
    output = 1
