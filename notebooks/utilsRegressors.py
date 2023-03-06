import pandas as pd
import numpy as np


def filter_array(array, w):
    filtered_array = pd.Series(array).rolling(w, center=True).mean()
    filtered_array[filtered_array.isna()] = 0
    return np.array(filtered_array)


def compute_absolute_change(array, w):
    array_f = filter_array(array, w)
    output = [False]+[array_f[i]>array_f[i-1] for i in range(1,len(array_f))]
    return np.multiply(output, 1) # convert boolean into binary


def compute_pos_deriative(array, w):
    array_f = filter_array(array, w)
    output = np.gradient(array_f) # compute derivative
    output[output < 0] == 0 # remove negative values, not of interest here
    return output


def compute_power(freq, amp):
    return freq*amp


def compute_power_change(freq, amp, w):
    array = filer_array(compute_power(freq, amp), w)
    output = [False]+[array[i]>array[i-1] for i in range(1,len(array))]
    return np.multiply(output, 1) # convert boolean into binary