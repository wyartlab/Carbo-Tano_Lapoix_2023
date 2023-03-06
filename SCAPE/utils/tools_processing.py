import numpy as np
import matplotlib.pyplot as plt
from random import sample


def compute_distance_to_midline(p1, p2, p3):
    """
    Returns distance of point p3 to a line defined by points p1 and p2
    :param p1: tuple of floats
    :param p2: tuple of floats
    :param p3: tuple of floats
    :return: float
    """

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    d = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)
    return d


def check_cells_side(Exp, Cells, colors={'left': 'magenta', 'right':'cyan'}):
    for plane in Exp.suite2pData.keys():
        plt.figure()
        plt.title('Plane {}\n{}'.format(plane, colors))
        plt.imshow(Exp.suite2pData[plane]['ops']['meanImg'], cmap='Greys')
        for Cell in Cells:
            if Cell.plane == plane:
                plt.plot(Cell.y_pos, Cell.x_pos, 'o', color=colors[Cell.side])
        _ = plt.ginput()
        plt.close()


def build_mean_image(Exp):
    planes = list(Exp.suite2pData.keys()).copy()
    planes.sort()
    limits_crop = Exp.limits_crop
    lim_sup = np.max(limits_crop['x_lim'])
    y_size = Exp.suite2pData[planes[0]]['ops']['meanImg'].shape[1]
    arrays = np.zeros((lim_sup, y_size, len(planes)))
    for i, plane in enumerate(planes):
        lims = list(limits_crop.loc[limits_crop['Unnamed: 0'] == plane, 'x_lim'])
        arrays[lims[0]:lims[1],:,i] = Exp.suite2pData[plane]['ops']['meanImg']
    output = np.mean(arrays, axis=2)
    return output
