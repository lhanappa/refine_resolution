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

# get generator model
filepath = './saved_model/gen_model'
#Generator = model.make_generator_model(len_high_size=40, scale=4)
#file_path = './saved_model/gen_model'
#Generator.save(file_path, overwrite=True, include_optimizer=False)

#Generator = tf.keras.models.load_model(filepath)
if os.path.exists(filepath):
    Generator = tf.keras.models.load_model(filepath)
    Generator.get_layer('dsd_x8').build(input_shape=(1, 40, 40, 1))
    Generator.get_layer('dsd_x4').build(input_shape=(1, 40, 40, 1))
    Generator.get_layer('dsd_x2').build(input_shape=(1, 40, 40, 1))
    Generator.get_layer('r1c_x8').build(input_shape=(1, 5, 5, 256))
    Generator.get_layer('r1c_x4').build(input_shape=(1, 10, 10, 512))
    Generator.get_layer('r1c_x2').build(input_shape=(1, 20, 20, 1024))
    Generator.get_layer('r1e_x8').build(input_shape=(1, 5, 5, 256))
    Generator.get_layer('r1e_x4').build(input_shape=(1, 10, 10, 512))
    Generator.get_layer('r1e_x2').build(input_shape=(1, 20, 20, 1024))
    Generator.get_layer('usc_x8').build(input_shape=(1, 5, 5, 32))
    Generator.get_layer('usc_x4').build(input_shape=(1, 10, 10, 144))
    Generator.get_layer('usc_x2').build(input_shape=(1, 20, 20, 320))
    print('load model')
else:
    Generator = model.make_generator_model(len_high_size=40, scale=4)
#Generator.optimizer=[tf.keras.optimizers.Adam, tf.keras.optimizers.Adam]
print(Generator)
Generator.save(filepath)
"""# data from ftp://cooler.csail.mit.edu/coolers/hg19/
name = './data/raw/Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
#name = 'Rao2014-K562-MboI-allreps-filtered.500kb.cool'
c = cooler.Cooler(name)
resolution = c.binsize
mat = c.matrix(balance=True).fetch('chr22')

[Mh, _] = operations.remove_zeros(mat)
Mh = Mh[0:40, 0:40]
print('MH: ', Mh.shape)

scale = 4
Ml = operations.sampling_hic(Mh, scale**2, fix_seed=True)
print('ML: ', Ml.shape)

# Normalization
# the input should not be type of np.matrix!
Ml = normalization.SCN_normalization(np.asarray(Ml), max_iter=3000)
Mh = normalization.SCN_normalization(np.asarray(Mh), max_iter=3000)

len_size = 40
hic_hr, index_1d_2d, _ = operations.divide_pieces_hic(
    Mh, block_size=len_size, save_file=False)
hic_hr = np.asarray(hic_hr)
print('shape hic_hr: ', hic_hr.shape)
hic_lr, _, _ = operations.divide_pieces_hic(
    Ml, block_size=len_size, save_file=False)
hic_lr = np.asarray(hic_lr)
print('shape hic_lr: ', hic_lr.shape)
'''with np.load('./datasets_hic.npz', allow_pickle=True) as data:
    a = data['hic']
    b = data['index_1D_2D']
    c = data['index_2D_1D']
    h = merge_hic(a, b)'''

true_hic_hr = hic_hr
print(true_hic_hr.shape)
[_, _, _, predict_hic_hr, _, _, _] = Generator(hic_lr[..., np.newaxis], training=False)
predict_hic_hr = np.squeeze(predict_hic_hr.numpy(), axis=3)
print(predict_hic_hr.shape)

predict_hic_hr_merge = operations.merge_hic( predict_hic_hr, index_1D_2D=index_1d_2d)
predict_hic_hr_merge = normalization.SCN_normalization(predict_hic_hr_merge, max_iter=3000)
print('merge predict hic hr', predict_hic_hr_merge.shape)

# chrop Mh
residual = Mh.shape[0] % int(len_size/2)
if residual > 0:
    Mh = Mh[0:-residual, 0:-residual]

diag = np.ones(Mh.shape) - np.diag(np.ones(Mh.shape[0])) - np.diag(np.ones(Mh.shape[0]-1), k=1) - np.diag(np.ones(Mh.shape[0]-1), k=-1)
Mh = Mh*diag
predict_hic_hr_merge = predict_hic_hr_merge*diag
print('sum Mh:', np.sum(np.abs(Mh)))
print('sum merge:', np.sum(np.abs(predict_hic_hr_merge)))
diff = np.abs(Mh-predict_hic_hr_merge)
print('sum diff: {:.5}'.format(np.sum(diff**2)))

hr_file = './data/input/Dixon2012_10kb/HR/chr22/Dixon2012_10kb_HR_chr22_0-2047.npz'
hr_hic = np.load(hr_file, allow_pickle=True)
hr_hic = hr_hic['hic']

fig, axs = plt.subplots(1, 3, figsize=(8, 15))
ax = axs[0].imshow(np.log1p(1000*predict_hic_hr_merge), cmap='RdBu_r')
axs[0].set_title('predict')
ax = axs[1].imshow(np.log1p(1000*Mh), cmap='RdBu_r')
axs[1].set_title('true')
ax = axs[2].imshow(np.log1p(1000*np.squeeze(hr_hic[0,:,:])), cmap='RdBu_r')
axs[2].set_title('true')
plt.tight_layout()
plt.show()

predict_hic_coo = operations.dense2tag(predict_hic_hr_merge)
true_hic_coo = operations.dense2tag(Mh)
print('shape coo predict hic: {}'.format(predict_hic_coo.shape))
print('shape coo predict hic: {}'.format(true_hic_coo.shape))"""
