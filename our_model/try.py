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

    Ml = operations.sampling_hic(Mh, scale**2, fix_seed=True)
    print('ML: ', Ml.shape)

    # Normalization
    # the input should not be type of np.matrix!
    Ml = np.asarray(Ml)
    Mh = np.asarray(Mh)
    Ml, Dl = operations.scn_normalization(Ml, max_iter=3000)
    print('Dl shape:{}'.format(Dl.shape))
    Mh, Dh = operations.scn_normalization(Mh, max_iter=3000)
    print('Dl shape:{}'.format(Dl.shape))


    if genomic_distance is None:
        max_boundary = None
    else:
        max_boundary = np.ceil(genomic_distance/(resolution))
    hic_hr, index_1d_2d, index_2d_1d = operations.divide_pieces_hic( Mh, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_hr = np.asarray(hic_hr, dtype=np.float32)
    print('shape hic_hr: ', hic_hr.shape)

    hic_lr, _, _ = operations.divide_pieces_hic( Ml, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_lr = np.asarray(hic_lr, dtype=np.float32)
    print('shape hic_lr: ', hic_lr.shape)

    true_hic_hr = hic_hr
    print('shape true hic_hr: ', true_hic_hr.shape)


    true_hic_hr_merge = operations.merge_hic( true_hic_hr, index_1D_2D=index_1d_2d, max_distance=max_boundary)
    print('shape of merge true hic hr', true_hic_hr_merge.shape)

    # chrop Mh
    residual = Mh.shape[0] % int(len_size/2)
    print('residual: {}'.format(residual))
    if residual > 0:
        Mh = Mh[0:-residual, 0:-residual]
        # true_hic_hr_merge = true_hic_hr_merge[0:-residual, 0:-residual]
        Dh = Dh[0:-residual]
        Dl = Dl[0:-residual]

    # recover M from scn to origin
    # Mh = operations.scn_recover(Mh, Dh)
    # true_hic_hr_merge = operations.scn_recover(true_hic_hr_merge, Dh)

    # remove diag and off diag
    k = max_boundary.astype(int)
    Mh = operations.filter_diag_boundary(Mh, diag_k=2, boundary_k=k)
    true_hic_hr_merge = operations.filter_diag_boundary(true_hic_hr_merge, diag_k=2, boundary_k=k)

    print('sum Mh:', np.sum(np.abs(Mh)))
    print('sum merge:', np.sum(np.abs(true_hic_hr_merge)))
    diff = np.abs(Mh-true_hic_hr_merge)
    print('sum diff: {:.5}'.format(np.sum(diff**2)))

    directory_sr = os.path.join(path, sr_path, sr_file, 'SR')
    compact = idx[0:-residual]

    directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)
    file = 'true_chr{}_{}.npz'.format(chromosome, resolution)
    np.savez_compressed(os.path.join(directory_sr, file), hic=Mh, compact=compact)
    print('Saving file:', file)
    file = 'truemerge_chr{}_{}.npz'.format(chromosome, resolution)
    np.savez_compressed(os.path.join(directory_sr, file), hic=true_hic_hr_merge, compact=compact)
    print('Saving file:', file)



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
            genomic_distance=2000000, start=0, end=None, draw_out=True)
