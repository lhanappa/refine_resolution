import glob
import imageio
import time
import datetime
from IPython import display
import matplotlib.pyplot as plt
from iced import normalization
import cooler
import numpy as np
import copy
import os
import sys
import shutil
from . import model
from .utils.operations import sampling_hic
from .utils.operations import divide_pieces_hic, merge_hic
from .utils.operations import redircwd_back_projroot
import tensorflow as tf

tf.keras.backend.set_floatx('float32')


# data from ftp://cooler.csail.mit.edu/coolers/hg19/

def run(train_data, test_data, len_size, scale, EPOCHS, summary=False):
    # get generator model
    Gen = model.make_generator_model(len_high_size=len_size, scale=scale)
    file_path = os.path.join(
        './our_model/saved_model/gen_model_'+str(len_size), 'gen_weights')
    if os.path.exists(file_path):
        pass
        # Gen.load_weights(file_path)

    # get discriminator model
    Dis = model.make_discriminator_model(len_high_size=len_size, scale=scale)
    file_path = os.path.join(
        './our_model/saved_model/dis_model_'+str(len_size), 'dis_weights')
    if os.path.exists(file_path):
        pass
        # Dis.load_weights(file_path)

    if summary:
        print(Gen.summary())
        tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
        print(Dis.summary())
        tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)

    model.train(Gen, Dis, train_data, EPOCHS, len_size, scale, test_data)

    file_path = os.path.join(
        './our_model/saved_model/gen_model_'+str(len_size), 'gen_weights')
    Gen.save_weights(file_path)

    file_path = os.path.join(
        './our_model/saved_model/dis_model_'+str(len_size), 'dis_weights')
    Dis.save_weights(file_path)


if __name__ == '__main__':
    # the size of input
    len_size = int(sys.argv[1])  # 40, 128, 200
    scale = 4
    # genomic_disstance is used for input path, nothing to do with model
    genomic_distance = int(sys.argv[2])  # 2000000, 2560000
    EPOCHS = 800
    BATCH_SIZE = 9
    root_path = redircwd_back_projroot(project_name='refine_resolution')
    data_path = os.path.join(root_path, 'data')
    #raw_path = 'raw'
    raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    input_path = '_'.join(
        ['input', 'ours', str(genomic_distance), str(len_size)])
    input_file = raw_hic.split('-')[0] + '_' + raw_hic.split('.')[1]
    #output_path = 'output'
    #output_file = input_file

    #['1', '2', '3', '4', '5','6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    #['16', '17', '18', '19', '20', '21', '22', 'X']
    chromosome_list = ['1', '2', '3', '4', '5', '6', '7',
                       '8', '9', '10', '11', '12', '13', '14', '15']
    hr_file_list = []

    for chri in chromosome_list:
        path = os.path.join(data_path, input_path,
                            input_file, 'HR', 'chr'+chri)
        if not os.path.exists(path):
            continue
        for file in os.listdir(path):
            if file.endswith(".npz"):
                pathfile = os.path.join(path, file)
                hr_file_list.append(pathfile)
    hr_file_list.sort()

    hic_hr = None
    hic_lr = None
    for hr_file in hr_file_list:
        lr_file = hr_file.replace('HR', 'LR')
        print(hr_file, lr_file)
        if (not os.path.exists(hr_file)) or (not os.path.exists(lr_file)):
            continue
        with np.load(hr_file, allow_pickle=True) as data:
            if hic_hr is None:
                hic_hr = data['hic']
            else:
                hic_hr = np.concatenate((hic_hr, data['hic']), axis=0)
        with np.load(lr_file, allow_pickle=True) as data:
            if hic_lr is None:
                hic_lr = data['hic']
            else:
                hic_lr = np.concatenate((hic_lr, data['hic']), axis=0)

    hic_lr = np.asarray(hic_lr).astype(np.float32)
    hic_hr = np.asarray(hic_hr).astype(np.float32)
    train_data = tf.data.Dataset.from_tensor_slices(
        (hic_lr[..., np.newaxis], hic_hr[..., np.newaxis])).batch(BATCH_SIZE)
    test_data = tf.data.Dataset.from_tensor_slices(
        (hic_lr[0:9, ..., np.newaxis], hic_hr[0:9, ..., np.newaxis])).batch(BATCH_SIZE)
    run(train_data=train_data, test_data=test_data,
        len_size=len_size, scale=scale, EPOCHS=EPOCHS, summary=False)
