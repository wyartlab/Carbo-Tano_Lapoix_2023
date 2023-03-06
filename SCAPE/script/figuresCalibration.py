import multipagetiff as mtif
import matplotlib.pyplot as plt
import numpy as np

# paths_in = {'before_deconvolv': '/media/mathilde.lapoix/SCAPE_MARCO/SCAPE/data/220118/unCorrected_F2_run2_HR.tiff',
#             'after_deconvolv': '/media/mathilde.lapoix/SCAPE_MARCO/SCAPE/data/220118/unCorrected_F2_run2_HR_deconvolved.tiff'}
#
# paths_out = {'before_deconvolv': '/media/mathilde.lapoix/Seagate Expansion Drive/220128_F2_run2_HR_before_dec',
#             'after_deconvolv': '/media/mathilde.lapoix/Seagate Expansion Drive/220128_F2_run2_HR_after_dec'}

paths_in = {'before_deconvolv': '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/220127/220127_F4_run2/deconvolution/unCorrected_220127_F4_run2_t396 (1).tiff',
            'after_deconvolv': '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/220127/220127_F4_run2/deconvolution/deconvolved_unCorrected_220127_F4_run2_t396.tiff'}

paths_out = {'before_deconvolv': '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/220127/220127_F4_run2/deconvolution/before_dec',
            'after_deconvolv': '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/SCAPE/220127/220127_F4_run2/deconvolution/after_dec'}

for i in paths_in.keys():
    stack = mtif.read_stack(paths_in[i])
    stack_arr = stack.pages[:,:,0:200]
    stack = mtif.Stack(stack_arr)

    plt.figure()
    mtif.plot_flatten(stack)
    plt.savefig(paths_out[i]+'_depthCoded.svg')

    plt.figure()
    s1 = stack.copy()
    s1.crop_horizontal = (70,120)
    s1.crop_vertical = (350,390)

    # Z limits
    s1.start_page = 80
    s1.end_page = 100

    plt.subplot(1,2,1)
    mtif.plot_selection(s1)
    plt.subplot(1,2,2)
    b = np.max(s1.pages, axis=0)
    plt.imshow(b)
    plt.colorbar()
    plt.savefig(paths_out[i]+'_zoom.svg')
