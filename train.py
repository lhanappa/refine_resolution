import model
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
import shutil
from utils.operations import sampling_hic
from utils.operations import divide_pieces_hic, merge_hic
import tensorflow as tf
tf.keras.backend.set_floatx('float32')


"""# data from ftp://cooler.csail.mit.edu/coolers/hg19/
name = 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
#name = 'Rao2014-K562-MboI-allreps-filtered.500kb.cool'
c = cooler.Cooler(name)
resolution = c.binsize
mat = c.matrix(balance=True).fetch('chr10')
idxy = ~np.all(np.isnan(mat), axis=0)
M = mat[idxy, :]
Mh = M[:, idxy]
#Mh = np.asarray(Mh[0:256, 0:256])
print('MH: ', Mh.shape)

scale = 4
img_l = sampling_hic(Mh, scale**2, fix_seed=True)
Ml = np.asarray(img_l)
print('ML: ', Ml.shape)

# Normalization
# the input should not be type of np.matrix!
Ml = normalization.SCN_normalization(Ml, max_iter=3000)
Mh = normalization.SCN_normalization(Mh, max_iter=3000)

len_size = 128

hic_lr,_,_ = divide_pieces_hic(Ml, block_size=len_size, save_file=False)
hic_hr,_,_ = divide_pieces_hic(Mh, block_size=len_size, save_file=False)
hic_lr = np.asarray(hic_lr)
hic_hr = np.asarray(hic_hr)"""


def run(train_data, test_data, len_size, scale, EPOCHS, summary=False):
    # get generator model
    Gen = model.make_generator_model(len_high_size=len_size, scale=scale)

    filepath = './saved_model/gen_model/gen_weights'
    if os.path.exists(filepath):
        Gen.load_weights(filepath)

    # get discriminator model
    Dis = model.make_discriminator_model(len_high_size=len_size, scale=scale)
    filepath = './saved_model/dis_model/dis_weights'
    if os.path.exists(filepath):
        Dis.load_weights(filepath)

    if summary:
        print(Gen.summary())
        tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
        print(Dis.summary())
        tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)

    model.train(Gen, Dis, train_data, EPOCHS, len_size, scale, test_data)

    file_path = './saved_model/gen_model/gen_weights'
    Gen.save_weights(file_path)
    #tf.keras.models.save_model(Gen, file_path, overwrite=False, include_optimizer=False)
    #Gen.save(file_path, overwrite=False, include_optimizer=False)
    file_path = './saved_model/dis_model/dis_weights'
    Dis.save_weights(file_path)
    #tf.keras.models.save_model(Dis, file_path, overwrite=False, include_optimizer=False)


if __name__ == '__main__':
    len_size = 40
    scale = 4
    genomic_distance=2000000
    EPOCHS = 600
    BATCH_SIZE = 9
    data_path = './data'
    raw_path = 'raw'
    raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    input_path =  '_'.join(['input', 'ours', str(genomic_distance), str(len_size)]) 
    input_file = raw_hic.split('-')[0] + '_' + raw_hic.split('.')[1]
    output_path = 'output'
    output_file = input_file
    #'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' 'X'
    chromosome_list = ['13', '14', '15']
    hr_file_list = []

    for chri in chromosome_list:
        path = os.path.join(data_path, input_path,
                            input_file, 'HR', 'chr'+chri)
        for file in os.listdir(path):
            if file.endswith(".npz"):
                pathfile = os.path.join(path, file)
                hr_file_list.append(pathfile)

    for hr_file in hr_file_list:
        print(hr_file)
        with np.load(hr_file, allow_pickle=True) as data:
            hic_hr = data['hic']
        lr_file = hr_file.replace('HR', 'LR')
        print(lr_file)
        with np.load(lr_file, allow_pickle=True) as data:
            hic_lr = data['hic']

        hic_lr = np.asarray(hic_lr).astype(np.float32)
        hic_hr = np.asarray(hic_hr).astype(np.float32)
        train_data = tf.data.Dataset.from_tensor_slices(
            (hic_lr[..., np.newaxis], hic_hr[..., np.newaxis])).batch(BATCH_SIZE)
        test_data = tf.data.Dataset.from_tensor_slices(
            (hic_lr[0:9, ..., np.newaxis], hic_hr[0:9, ..., np.newaxis])).batch(BATCH_SIZE)
        run(train_data=train_data, test_data=test_data,
            len_size=len_size, scale=scale, EPOCHS=EPOCHS, summary=False)
