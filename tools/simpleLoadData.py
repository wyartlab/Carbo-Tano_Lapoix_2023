import numpy as np
import pandas as pd
import shelve

fishlabel = '210121_F05'
plane = '70um_bh'
output_path = '/network/lustre/iss01/wyart/analyses/mathilde.lapoix/MLR/Calcium_Imaging/' + fishlabel + '/' + plane

shelf_in = shelve.open(output_path + '/shelve_calciumAnalysis.out')
for key in shelf_in:
    globals()[key] = shelf_in[key]
shelf_in.close()
