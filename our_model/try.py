import time
import datetime
import matplotlib.pyplot as plt
import cooler
import numpy as np
import copy
import os

import model
from utils import operations
# import tensorflow as tf
# tf.keras.backend.set_floatx('float32')

# 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
# data from ftp://cooler.csail.mit.edu/coolers/hg19/


def predict(path='./data',
            raw_path='raw',
            raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
            model_path=None,
            sr_path='output',
            chromosome='22',
            scale=4,
            len_size=200,
            genomic_distance=2000000,
            start=None, end=None, draw_out=False):
    sr_file = raw_file.split('-')[0] + '_' + raw_file.split('-')[1] + '_' + raw_file.split('.')[1]
    directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)
    if not os.path.exists(directory_sr):
        os.makedirs(directory_sr)


    name = os.path.join(path, raw_path, raw_file)
    c = cooler.Cooler(name)
    resolution = c.binsize
    mat = c.matrix(balance=True).fetch('chr'+chromosome)
    [Mh, idx] = operations.remove_zeros(mat)
    print('shape HR: ', Mh.shape)

    if start is None:
        start = 0
    if end is None:
        end = Mh.shape[0]

    Mh = Mh[start:end, start:end]
    print('MH: ', Mh.shape)

    # Normalization
    # the input should not be type of np.matrix!
    Mh = np.asarray(Mh)
    # Mh, Dh = operations.scn_normalization(Mh, max_iter=3000)


    if genomic_distance is None:
        max_boundary = None
    else:
        max_boundary = np.ceil(genomic_distance/(resolution))
    hic_hr, index_1d_2d, index_2d_1d = operations.divide_pieces_hic( Mh, block_size=len_size, max_distance=max_boundary, save_file=False)
    # hic_hr = np.asarray(hic_hr, dtype=np.float32)
    true_hic_hr = hic_hr
    true_hic_hr_merge = operations.merge_hic( true_hic_hr, index_1D_2D=index_1d_2d, max_distance=max_boundary)
    print('shape of merge true hic hr', true_hic_hr_merge.shape)

    # chrop Mh
    residual = Mh.shape[0] % int(len_size/2)
    print('residual: {}'.format(residual))
    if residual > 0:
        Mh = Mh[0:-residual, 0:-residual]
        # Dh = Dh[0:-residual]

    # recover M from scn to origin
    # Mh = operations.scn_recover(Mh, Dh)
    # true_hic_hr_merge = operations.scn_recover(true_hic_hr_merge, Dh)

    # remove diag and off diag
    # k = max_boundary.astype(int)
    # Mh = operations.filter_diag_boundary(Mh, diag_k=2, boundary_k=k)
    # true_hic_hr_merge = operations.filter_diag_boundary(true_hic_hr_merge, diag_k=2, boundary_k=k)

    print('sum Mh:', np.sum(np.abs(Mh)))
    print('sum merge:', np.sum(np.abs(true_hic_hr_merge)))
    diff = np.abs(Mh-true_hic_hr_merge)
    print('sum diff: {:.5}'.format(np.sum(diff**2)))




if __name__ == '__main__':
    root = operations.redircwd_back_projroot(project_name='refine_resolution')
    data_path = os.path.join(root, 'data')
    max_dis = 2000000
    len_size = 200
    predict(path=data_path,
            raw_path='raw',
            raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
            chromosome='22',
            scale=4,
            len_size=200,
            sr_path='_'.join(['output', 'ours', str(max_dis), str(len_size)]),
            genomic_distance=max_dis, start=0, end=None, draw_out=True)
