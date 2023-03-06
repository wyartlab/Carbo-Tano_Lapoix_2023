from suite2p.extraction import dcnv
import numpy as np
import sys
import os
import logging


# extract from command line

output_path = sys.argv[1]
dff_name = sys.argv[2]
fishlabel = str(sys.argv[3])
trial = str(sys.argv[4])

# initialise


script = os.path.basename(__file__)
handlers = [logging.FileHandler(output_path + fishlabel + '/' + trial + '/logs/' + script + '.log'), logging.StreamHandler()]
logging.basicConfig(handlers=handlers,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# define params for spike deconv

tau = 2 # timescale of indicator
fs = 9.965 # sampling rate in Hz
# for computing and subtracting baseline
baseline = 'maximin' # take the running max of the running min after smoothing with gaussian
sig_baseline = 10.0 # in bins, standard deviation of gaussian with which to smooth
win_baseline = 20.0 # in seconds, window in which to compute max/min filters

ops = {'tau': tau, 'fs': fs, 'batch_size': 100,
       'baseline': baseline, 'sig_baseline': sig_baseline, 'win_baseline': win_baseline}

# load traces and subtract neuropil
DFF = np.load(output_path + fishlabel + '/' + trial + '/dataset/' + dff_name)

# get spikes
spks = dcnv.oasis(F=DFF, ops=ops)

np.save(output_path + fishlabel + '/' + trial + '/dataset/spks_from_' + dff_name, spks)
