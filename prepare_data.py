import numpy as np
import os
import sys
import cooler

from iced import normalization
from utils import operations

def configure():
    # set path
    data_path = './data'
    raw_path = 'raw'
    raw_hic = 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
    input_path = 'input'
    input_file = raw_hic.split('-')[0] + '_' + raw_hic.split('.')[1]
    output_path = 'output'
    output_file = input_file

    resolution = None  # assigned by cooler binsizes
    scale = 4
    len_size = 40
    block_size = 2048  # number of entries in one file

    # load raw hic matrix
    file = os.path.join(data_path, raw_path, raw_hic)
    print(file)
    cool_hic = cooler.Cooler(file)
    resolution = cool_hic.binsize
    return cool_hic, resolution, scale, len_size, \
            block_size, data_path, \
            [raw_path, raw_hic], \
            [input_path, input_file], \
            [output_path, output_file]


def save_samples(configure=None, chromosome=None):
    cool_hic, resolution, scale, len_size, block_size, data_path, \
    [raw_path, raw_hic], \
    [input_path, input_file], \
    [output_path, output_file] = configure
    chromosome = 'chr' + chromosome
    mat = cool_hic.matrix(balance=True).fetch(chromosome)
    Mh, _ = operations.remove_zeros(mat)
    #Mh = Mh[0:512, 0:512]
    print('MH: ', Mh.shape)
    Ml = operations.sampling_hic(Mh, scale**2, fix_seed=True)
    print('ML: ', Ml.shape)

    # Normalization
    # the input should not be type of np.matrix!
    Ml = normalization.SCN_normalization(np.asarray(Ml), max_iter=3000)
    Mh = normalization.SCN_normalization(np.asarray(Mh), max_iter=3000)

    hic_hr, index_1d_2d, index_2d_1d = operations.divide_pieces_hic(
        Mh, block_size=len_size, save_file=False)
    hic_hr = np.asarray(hic_hr)
    print('shape hic_hr: ', hic_hr.shape)
    hic_lr, _, _ = operations.divide_pieces_hic(
        Ml, block_size=len_size, save_file=False)
    hic_lr = np.asarray(hic_lr)
    print('shape hic_lr: ', hic_lr.shape)
    directory_hr = os.path.join(
        data_path, input_path, input_file, 'HR', chromosome)
    directory_lr = os.path.join(
        data_path, input_path, input_file, 'LR', chromosome)
    directory_sr = os.path.join(
        data_path, output_path, output_file, 'SR', chromosome)
    if not os.path.exists(directory_hr):
        os.makedirs(directory_hr)
    if not os.path.exists(directory_lr):
        os.makedirs(directory_lr)
    if not os.path.exists(directory_sr):
        os.makedirs(directory_sr)

    for ibs in np.arange(0, hic_hr.shape[0], block_size):
        start = ibs
        end = min(start+block_size, hic_hr.shape[0])

        hic_m = hic_hr[start:end, :, :]
        pathfile = input_file+'_HR_' + chromosome + \
            '_' + str(start) + '-' + str(end-1)
        pathfile = os.path.join(directory_hr, pathfile)
        print(pathfile)
        np.savez_compressed(pathfile+'.npz', hic=hic_m, index_1D_2D=index_1d_2d, index_2D_1D=index_2d_1d)

        hic_m = hic_lr[start:end, :, :]
        pathfile = input_file+'_LR_' + chromosome + \
            '_' + str(start) + '-' + str(end-1)
        pathfile = os.path.join(directory_lr, pathfile)
        print(pathfile)
        np.savez_compressed(pathfile+'.npz', hic=hic_m, index_1D_2D=index_1d_2d, index_2D_1D=index_2d_1d)


if __name__ == '__main__':
    config = configure()
    chromosome_list = [str(sys.argv[1])]
    for chri in chromosome_list:
        save_samples(config, chromosome=chri)