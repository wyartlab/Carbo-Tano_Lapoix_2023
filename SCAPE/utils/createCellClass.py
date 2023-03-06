import pandas as pd
import numpy as np
from utils.tools_processing import compute_distance_to_midline


def calc_DFF(f, baseline):
    f0 = np.nanmedian(f[baseline[0]:baseline[1]])
    dff = [(i - f0) / f0 for i in range(len(f))]
    return dff


class Cell:
    dff = []

    def __init__(self, exp, plane, cell, id):

        self.cellID = id
        self.init_cellID = cell
        self.x_pos = exp.suite2pData[plane]['stat'][cell]['med'][1]
        self.y_pos = exp.suite2pData[plane]['stat'][cell]['med'][0]
        self.norm_y_pos = np.nan
        self.plane = int(plane)
        self.group = np.nan
        self.side = np.nan
        self.distance_to_midline = np.nan
        self.F_corrected = exp.suite2pData[plane]['F_corrected'][cell,]
        # self.spks = spks[plane][cell,]
        self.spks = exp.suite2pData[plane]['spks'][cell,]
        # self.dff = dff[plane][cell,]
        self.dff = exp.suite2pData[plane]['dff'][cell,]
        self.dff_corrected = exp.suite2pData[plane]['dff'][cell,]
        # self.noise = noise[plane][cell]
        self.noise = exp.suite2pData[plane]['noise'][cell]

    def compute_dff(self, baseline):
        f0 = np.nanmedian(self.F_corrected[baseline[0]:baseline[1]])
        self.dff = [(i - f0) / f0 for i in range(len(self.F_corrected))]

    def mask_and_fill_signal(self, Exp):
        bad_frames = Exp.bad_frames
        self.dff_corrected[bad_frames] = np.nan
        self.dff_corrected = np.array(pd.Series(self.dff_corrected).interpolate())

    def assign_cell_group(self, dict_x, dorsoventral_limit):
        x = self.x_pos
        z = self.plane
        for group, limits in dict_x.items():
            if limits[0] < x <= limits[1]:
                output = group
            else:
                continue
        if output in ['retropontine', 'pontine', 'prepontine']:
            if z < dorsoventral_limit:
                output = output + '_dorsal'
            else:
                output = output + '_ventral'
        self.group = output

    def assign_side(self, Exp):
        z = self.plane

        distance = compute_distance_to_midline(p1=Exp.midline_lim[z][0],
                                               p2=Exp.midline_lim[z][1],
                                               p3=(self.x_pos, self.y_pos))
        if distance > 0:
            output = 'right'
        else:
            output = 'left'

        self.side = output
        self.distance_to_midline = distance

    def compute_norm_y_pos(self, Exp):

        norm_factor = list(Exp.limits_crop[Exp.limits_crop['Unnamed: 0'] == self.plane].x_lim)[0]
        self.norm_y_pos = self.y_pos+norm_factor

