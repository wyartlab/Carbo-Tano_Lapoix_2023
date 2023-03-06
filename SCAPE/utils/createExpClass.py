import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from utils.processSuite2pOutput import correct_suite2p_outputs


class Run:

    def __init__(self, summary, exp_id):
        self.fishID = summary.fishlabel[exp_id]
        self.runID = summary.run[exp_id]
        self.date = summary.date[exp_id]
        self.calciumIndicator = summary.date[exp_id]
        self.stage = summary.stage[exp_id]
        self.enucleated = summary.enucleated[exp_id]
        self.frameRateSCAPE = summary.frameRateSCAPE[exp_id]
        self.frameRateBeh = summary.frameRateBeh[exp_id]
        self.laserPower = summary.laserPower[exp_id]
        self.suite2pAnalysis = summary.suite2pAnalysis[exp_id]
        self.suite2pPath = summary.suite2pPath[exp_id]
        self.savePath = summary.savePath[exp_id]
        self.bad_frames = [int(i) for i in summary.bad_frames[exp_id].split(',')]

        self.nPlanesAnalysed = np.nan
        self.suite2pData = dict()

    def load_behavior_df(self):
        df_bout = pd.read_pickle(self.savePath + '/df_bout.pkl')
        df_frame = pd.read_pickle(self.savePath + '/df_frame.pkl')

        return df_bout, df_frame

    def load_suite2p_outputs(self):
        if self.suite2pAnalysis:

            planesAnalysed = next(os.walk(self.suite2pPath))[1]

            for i in planesAnalysed:
                try:
                    self.suite2pData[int(i)] = dict()
                except ValueError:
                    i = i.split('_')[-1]
                try:
                    self.suite2pData[int(i)]['F'] = np.load(self.suite2pPath + i + '/suite2p/plane0/F.npy',
                                                            allow_pickle=True)
                    self.suite2pData[int(i)]['Fneu'] = np.load(self.suite2pPath + i +
                                                               '/suite2p/plane0/Fneu.npy', allow_pickle=True)
                    self.suite2pData[int(i)]['spks'] = np.load(self.suite2pPath + i +
                                                               '/suite2p/plane0/spks.npy', allow_pickle=True)
                    self.suite2pData[int(i)]['stat'] = np.load(self.suite2pPath + i +
                                                               '/suite2p/plane0/stat.npy', allow_pickle=True)
                    ops = np.load(self.suite2pPath + i + '/suite2p/plane0/ops.npy', allow_pickle=True)
                    self.suite2pData[int(i)]['ops'] = ops.item()
                    self.suite2pData[int(i)]['iscell'] = np.load(self.suite2pPath + i +
                                                                 '/suite2p/plane0/iscell.npy', allow_pickle=True)
                    print('succesfully loaded suite2p outputs for plane {} at location\n{}'.format(i,
                                                                                                   self.suite2pPath + i))

                    if not hasattr(self, 'nFramesSCAPE'):
                        setattr(self, 'nFramesSCAPE', self.suite2pData[int(i)]['F'].shape[1])

                except FileNotFoundError:
                    print('No outputs found at this path:\n')
                    print(self.suite2pPath + i)
                    self.suite2pData[int(i)] = None

        else:
            print('No suite2p analysis set for this run.')
            self.suite2pData = None

    def correct_suite2p_outputs(self, plane):

        F_corrected, cells = correct_suite2p_outputs(self.suite2pData[plane]['F'],
                                                     self.suite2pData[plane]['iscell'],
                                                     self.suite2pData[plane]['Fneu'])
        self.suite2pData[plane]['F_corrected'] = F_corrected
        self.suite2pData[plane]['cells'] = cells

    def filter_f(self, plane, window=3):
        filtered_f = np.zeros(self.suite2pData[plane]['F_corrected'].shape)
        for cell in self.suite2pData[plane]['cells']:
            trace = pd.Series(self.suite2pData[plane]['F_corrected'][cell])
            filtered_f[cell,] = np.array(trace.interpolate().rolling(window=window, center=True).median())
        self.suite2pData[plane]['F_corrected_filter'] = filtered_f

    def define_midline_per_plane(self):
        planes = list(self.suite2pData.keys()).copy()
        planes.sort()
        setattr(self, 'midline_lim', {})
        for plane in planes:
            plt.figure(figsize=(14, 10))
            plt.title(plane)
            vmax = np.percentile(self.suite2pData[plane]['ops']['meanImg'], 90)
            plt.imshow(self.suite2pData[plane]['ops']['meanImg'], cmap='Greys', vmax=vmax)
            try:
                plt.plot([lim1[0], lim2[0]], [lim1[1], lim2[1]], '--o')
            except UnboundLocalError:
                pass
            lim1, lim2 = plt.ginput(2)
            self.midline_lim[plane] = (lim1, lim2)
            plt.close()

    def build_mean_image(self):
        planes = list(self.suite2pData.keys()).copy()
        planes.sort()
        limits_crop = self.limits_crop
        lim_sup = np.max(limits_crop['x_lim'])
        y_size = self.suite2pData[planes[0]]['ops']['meanImg'].shape[1]
        arrays = np.zeros((lim_sup, y_size, len(planes)))
        for i, plane in enumerate(planes):
            lims = list(limits_crop.loc[limits_crop['Unnamed: 0'] == plane, 'x_lim'])
            arrays[lims[0]:lims[1], :, i] = self.suite2pData[plane]['ops']['meanImg']
        output = np.mean(arrays, axis=2)
        setattr(self, 'mean_background', output)

    # def build_mean_image_sagittal(self):
    #     planes = list(self.suite2pData.keys()).copy()
    #     planes.sort()
    #     limits_crop = self.limits_crop
    #     lim_sup = np.max(limits_crop['x_lim'])
    #     y_size = self.suite2pData[planes[0]]['ops']['meanImg'].shape[1]
    #     arrays = np.zeros((lim_sup, y_size, len(planes)))
    #     for i, plane in enumerate(planes):
    #         lims = list(limits_crop.loc[limits_crop['Unnamed: 0'] == plane, 'x_lim'])
    #         arrays[lims[0]:lims[1], :, i] = self.suite2pData[plane]['ops']['meanImg']
    #     output = np.mean(arrays, axis=1)
    #     setattr(self, 'mean_background', output)

    def assign_behavior_trace(self, df_frame):
        setattr(self, 'tail_angle', np.array(df_frame.Tail_angle).copy())

    def assign_time_indices(self):
        setattr(self, 'time_indices_bh', np.arange(len(self.tail_angle)) / self.frameRateBeh)
        setattr(self, 'time_indices_SCAPE', np.arange(self.nFramesSCAPE) / self.frameRateSCAPE)
