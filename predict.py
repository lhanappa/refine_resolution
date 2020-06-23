import time
import datetime
import matplotlib.pyplot as plt
from iced import normalization
import cooler
import numpy as np
import copy
import os

import model
from utils import operations
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

# 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
# data from ftp://cooler.csail.mit.edu/coolers/hg19/


def predict(path='./data',
            raw_path='raw',
            raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
            chromosome='22',
            scale=4,
            len_size=200,
            sr_path='output',
            genomic_distance = 2000000,
            start=None, end=None, draw_out=False):
    sr_file = raw_file.split('-')[0] + '_' + raw_file.split('.')[1]
    directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)
    if not os.path.exists(directory_sr):
        os.makedirs(directory_sr)

    # get generator model
    file_path = './saved_model/gen_model_' + str(len_size)+'/gen_weights'
    Generator = model.make_generator_model(len_high_size=len_size, scale=4)
    Generator.load_weights(file_path)
    print(Generator)

    name = os.path.join(path, raw_path, raw_file)
    c = cooler.Cooler(name)
    resolution = c.binsize
    mat = c.matrix(balance=True).fetch('chr'+chromosome)

    [Mh, _] = operations.remove_zeros(mat)
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
    Ml = normalization.SCN_normalization(np.asarray(Ml), max_iter=3000)
    Mh = normalization.SCN_normalization(np.asarray(Mh), max_iter=3000)

    if genomic_distance is None:
        max_boundary = None
    else:
        max_boundary = np.ceil(genomic_distance/(resolution))
    hic_hr, index_1d_2d, index_2d_1d = operations.divide_pieces_hic(
        Mh, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_hr = np.asarray(hic_hr, dtype=np.float32)
    print('shape hic_hr: ', hic_hr.shape)
    hic_lr, _, _ = operations.divide_pieces_hic(
        Ml, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_lr = np.asarray(hic_lr, dtype=np.float32)
    print('shape hic_lr: ', hic_lr.shape)

    true_hic_hr = hic_hr
    print('shape true hic_hr: ', true_hic_hr.shape)
    [_, _, predict_hic_hr, _, _] = Generator(
        hic_lr[..., np.newaxis], training=False)
    predict_hic_hr = np.squeeze(predict_hic_hr.numpy(), axis=3)
    print(predict_hic_hr.shape)

    sr_file += '_chr'+chromosome
    file_path = os.path.join(directory_sr, sr_file)
    np.savez_compressed(file_path+'.npz', predict_hic=predict_hic_hr,
                        true_hic=true_hic_hr, index_1D_2D=index_1d_2d, index_2D_1D=index_2d_1d, start_id=start, end_id=end)

    predict_hic_hr_merge = operations.merge_hic(
        predict_hic_hr, index_1D_2D=index_1d_2d, max_distance=max_boundary)
    #predict_hic_hr_merge = normalization.SCN_normalization(predict_hic_hr_merge, max_iter=3000)
    print('shape of merge predict hic hr', predict_hic_hr_merge.shape)

    # chrop Mh
    residual = Mh.shape[0] % int(len_size/2)
    if residual > 0:
        Mh = Mh[0:-residual, 0:-residual]

    k = max_boundary.astype(int)
    Mh = operations.filter_diag_boundary(Mh, diag_k=2, boundary_k=k)
    predict_hic_hr_merge = operations.filter_diag_boundary(
        predict_hic_hr_merge, diag_k=2, boundary_k=k)

    print('sum Mh:', np.sum(np.abs(Mh)))
    print('sum merge:', np.sum(np.abs(predict_hic_hr_merge)))
    diff = np.abs(Mh-predict_hic_hr_merge)
    print('sum diff: {:.5}'.format(np.sum(diff**2)))

    if draw_out:
        fig, axs = plt.subplots(1, 2, figsize=(8, 15))
        ax = axs[0].imshow(np.log1p(1000*predict_hic_hr_merge), cmap='RdBu_r')
        axs[0].set_title('predict')
        ax = axs[1].imshow(np.log1p(1000*Mh), cmap='RdBu_r')
        axs[1].set_title('true')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    predict(start=0, end=400, draw_out=True)
