import time
import datetime
import matplotlib.pyplot as plt
from iced import normalization
import cooler
import numpy as np
import copy

from utils import operations
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

# get generator model
time =  '20200525-121741'
filepath = './saved_model/'+time + '/gen_model'
Generator = tf.keras.models.load_model(filepath)

# data from ftp://cooler.csail.mit.edu/coolers/hg19/
name = 'Dixon2012-H1hESC-HindIII-allreps-filtered.100kb.cool'
#name = 'Rao2014-K562-MboI-allreps-filtered.500kb.cool'
c = cooler.Cooler(name)
resolution = c.binsize
mat = c.matrix(balance=True).fetch('chr1')

[Mh, _] = operations.remove_zeros(mat)
Mh = Mh[0:512, 0:512]
print('MH: ', Mh.shape)

scale = 4
Ml = operations.sampling_hic(Mh, scale**2, fix_seed=True)
print('ML: ', Ml.shape)

# Normalization
# the input should not be type of np.matrix!
Ml = normalization.SCN_normalization(Ml, max_iter=3000)
Mh = normalization.SCN_normalization(Mh, max_iter=3000)

len_size = 128
hic_hr, index_1d_2d, _ = operations.divide_pieces_hic(Mh, block_size=len_size, save_file=False)
hic_hr = np.asarray(hic_hr)
print('shape hic_hr: ', hic_hr.shape)
hic_lr, _, _ = operations.divide_pieces_hic(Ml, block_size=len_size, save_file=True)
hic_lr = np.asarray(hic_lr)
print('shape hic_lr: ', hic_lr.shape)
'''with np.load('./datasets_hic.npz', allow_pickle=True) as data:
    a = data['hic']
    b = data['index_1D_2D']
    c = data['index_2D_1D']
    h = merge_hic(a, b)'''

true_hic_hr = hic_hr
[_, _, _, predict_hic_hr, _, _, _] = Generator(
    hic_lr[..., np.newaxis], training=False)
print(true_hic_hr.shape)
print(predict_hic_hr.shape)

'''
BUFFER_SIZE = 1
BATCH_SIZE = 9
predict_data_lr = tf.data.Dataset.from_tensor_slices(hic_lr[..., np.newaxis]).batch(BATCH_SIZE)
true_hic_hr = tf.data.Dataset.from_tensor_slices(hic_hr[..., np.newaxis]).batch(BATCH_SIZE)
for img in predict_data_lr.take(2):
    [_,_,_,predict_hic_hr,_,_,_] = Generator(img, training=False)
    print(predict_hic_hr.shape)
'''

predict_hic_hr = list(np.squeeze(predict_hic_hr, axis=-1))
predict_hic_hr_merge = operations.merge_hic(predict_hic_hr, index_1D_2D=index_1d_2d)
print(predict_hic_hr_merge.shape)

print('sum Mh:', np.sum(np.abs(Mh)))
print('sum merge:', np.sum(np.abs(predict_hic_hr_merge)))
print('sum diff: {:.3}, rate {:.3}'.format(np.sum(np.abs(Mh-predict_hic_hr_merge)), np.sum(np.abs(Mh-predict_hic_hr_merge))/np.sum(np.abs(Mh))))
plt.figure(figsize=(5, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.log2(predict_hic_hr_merge), cmap='RdBu_r')
plt.subplot(1, 2, 2)
plt.imshow(np.log2(Mh), cmap='RdBu_r')
plt.tight_layout()
plt.show()
