from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.ndimage

# Load empty tiff to save the coordinates in

path = '/network/lustre/iss01/wyart/analyses/martin.carbotano/MLR/Anatomy_to_MLR/map_injected_NR_npiAtlas/registration_code_python_Faustine/coordinates_in_tiff/'

img = Image.open(path + "/MLR.tif")
images = []

for i in range(img.n_frames):
    img.seek(i)
    images.append(np.array(img))

img_array = np.array(images)
mirror_array = img_array.copy()

# Load coordinates of objects

df_cor = pd.read_csv(path + 'MLR_cell_coordinates.csv', index_col=0)

# Loop for each point, input coordinates into the corresponding array

for i in range(len(df_cor)):
    x = df_cor.loc[i + 1, 'x']
    y = df_cor.loc[i + 1, 'y']
    z = df_cor.loc[i + 1, 'plane']
    if df_cor.loc[i + 1, 'df'] == 'A':
        img_array[z, y, x] = 1
    else:
        mirror_array[z, y, x] = 1

# expand dots to surrounding pixels (square)
factor = 5
output = scipy.ndimage.morphology.binary_dilation(img_array, np.ones((factor, factor, factor))).astype(np.uint8)
output_mirror = scipy.ndimage.morphology.binary_dilation(mirror_array, np.ones((factor, factor, factor))).astype(
    np.uint8)

# save output tif as individuals png images
output[np.where(output == 1)] = 255
output_mirror[np.where(output_mirror == 1)] = 255

for i in range(img.n_frames):

    if len(str(i)) == 1:
        output_path = '/home/mathilde.lapoix/Documents/Analysis/mlr/alignment_MLR_pos/img_00' + str(i) + '.png'
        path2 = '/home/mathilde.lapoix/Documents/Analysis/mlr/alignment_MLR_pos_mirror/img_00' + str(i) + '.png'
    elif len(str(i)) == 2:
        output_path = '/home/mathilde.lapoix/Documents/Analysis/mlr/alignment_MLR_pos/img_0' + str(i) + '.png'
        path2 = '/home/mathilde.lapoix/Documents/Analysis/mlr/alignment_MLR_pos_mirror/img_0' + str(i) + '.png'
    else:
        output_path = '/home/mathilde.lapoix/Documents/Analysis/mlr/alignment_MLR_pos/img_' + str(i) + '.png'
        path2 = '/home/mathilde.lapoix/Documents/Analysis/mlr/alignment_MLR_pos_mirror/img_' + str(i) + '.png'

    Image.fromarray(output[i, :, :]).convert('L').save(output_path)
    Image.fromarray(output_mirror[i, :, :]).convert('L').save(path2)

#  TODO: build tiff ith all electrodes placement, color-coded by MLR pos or not

path = '/network/lustre/iss01/wyart/analyses/martin.carbotano/MLR/Anatomy_to_MLR/Projection_to_MLR_Max-Planck_Atlas/'

img = Image.open(path + "/MLR.tif")
images = []

for i in range(img.n_frames):
    img.seek(i)
    images.append(np.array(img))

img_array = np.array(images)
img_array[:] = 0

# Load coordinates of objects

df_elec = pd.read_csv('/network/lustre/iss01/wyart/analyses/mathilde.lapoix/'
                      'MLR/Behavior/analysis_10/df_electrode_placement.csv')
output_path = '/home/mathilde.lapoix/Documents/Analysis/MLR/electrode_placement_borders/'

# Loop for each point, input coordinates into the corresponding array

for i in df_elec[df_elec.position == 'MLR'].index:
    x = df_elec.loc[i, 'x']
    y = df_elec.loc[i, 'y']
    z = df_elec.loc[i, 'z']
    value = df_elec.loc[i, 'median_ratio_f_s']
    img_array[z, y, x] = int(((value + 1) / 2) * 255)


def sphere(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1, 2 * n + 1))
    x, y, z = np.indices((2 * n + 1, 2 * n + 1, 2 * n + 1))
    mask = (x - n) ** 2 + (y - n) ** 2 + (z - n) ** 2 <= n ** 2
    struct[mask] = 1
    return struct.astype(np.bool)


#  n = 5  # radius of a spherical structuring element in pixels

# Minkowski addition (dilation)

#  binary_dilation = scipy.ndimage.binary_dilation(img_array, structure=sphere(n)).astype(np.uint8)
# dilation = scipy.ndimage.grey_dilation(img_array, footprint=sphere(n)).astype(np.uint8)
# struct_grey_dilation = scipy.ndimage.grey_dilation(img_array, structure=sphere(n)).astype(np.uint8)
binary_dilation = scipy.ndimage.binary_dilation(img_array, structure=sphere(10)).astype(np.uint8)
binary_dilation_small = scipy.ndimage.binary_dilation(img_array, structure=sphere(7)).astype(np.uint8)

a = np.subtract(binary_dilation, binary_dilation_small)
a[np.where(a == 1)] = 255

op = output_path

for i in range(img.n_frames):

    if len(str(i)) == 1:
        output_path = op + 'img_00' + str(i) + '.png'
    elif len(str(i)) == 2:
        output_path = op + 'img_0' + str(i) + '.png'
    else:
        output_path = op + 'img_' + str(i) + '.png'
    Image.fromarray(a[i, :, :]).convert('L').save(output_path)

# dilation[np.where(dilation == 1)] = 255
#
# for i in range(img.n_frames):
#
#     if len(str(i)) == 1:
#         output_path = '/home/mathilde.lapoix/Documents/Analysis/MLR/electrode_placement/img_00' + str(i) + '.png'
#         path2 = '/home/mathilde.lapoix/Documents/Analysis/MLR/electrode_placement_2/img_00' + str(i) + '.png'
#     elif len(str(i)) == 2:
#         output_path = '/home/mathilde.lapoix/Documents/Analysis/MLR/electrode_placement/img_0' + str(i) + '.png'
#         path2 = '/home/mathilde.lapoix/Documents/Analysis/MLR/electrode_placement_2/img_0' + str(i) + '.png'
#     else:
#         output_path = '/home/mathilde.lapoix/Documents/Analysis/MLR/electrode_placement/img_' + str(i) + '.png'
#         path2 = '/home/mathilde.lapoix/Documents/Analysis/MLR/electrode_placement_2/img_' + str(i) + '.png'
#
#     Image.fromarray(dilation[i, :, :]).convert('L').save(path2)
#     Image.fromarray(struct_grey_dilation[i, :, :]).convert('L').save(output_path)
