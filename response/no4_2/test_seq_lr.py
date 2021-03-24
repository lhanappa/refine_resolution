# The authors use a random down-sampling procedure to generate the low-resolution Hi-C matrices.
# How well do these simulated low-resolution Hi-C matrices mimic experimentally-determined low-resolution Hi-C matrices?

# https://data.4dnucleome.org/experiment-set-replicates/4DNES3RHDBBR/#processed-files
# 4 replicates: 4DNEXNMDXSP1, 4DNEXYNM1II5, 4DNEXT3VJDNU, 4DNEXWGWFK1T
# combined: 4DNES3RHDBBR
import cooler
import sys, os
import numpy as np

def sampling_hic(hic_matrix, sampling_ratio, fix_seed=False):
    """sampling dense hic matrix"""
    m = np.matrix(hic_matrix)
    all_sum = m.sum(dtype='float')
    idx_prob = np.divide(m, all_sum, out=np.zeros_like(m), where=all_sum != 0)
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],)))
    idx_prob = np.squeeze(idx_prob)
    sample_number_counts = int(all_sum/(2*sampling_ratio))
    id_range = np.arange(m.shape[0]*m.shape[1])
    if fix_seed:
        np.random.seed(0)
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob)
    sample_m = np.zeros_like(m)  
    for i in np.arange(sample_number_counts):
        x = int(id_x[i]/m.shape[0])
        y = int(id_x[i] % m.shape[0])
        sample_m[x, y] += 1.0
    sample_m = np.transpose(sample_m) + sample_m
    return np.asarray(sample_m)

def load_hic(path, name):
    hic = cooler.Cooler(os.paht.join(path, name))
    return hic