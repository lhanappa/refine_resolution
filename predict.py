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

path = './data'
raw_path = 'raw'
raw_file = 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
chromosome = '22'

sr_path = 'output'
sr_file = raw_file.split('-')[0] + '_' + raw_file.split('.')[1]
directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)
if not os.path.exists(directory_sr):
    os.makedirs(directory_sr)

# get generator model
file_path = './saved_model/gen_model/gen_weights'
Generator = model.make_generator_model(len_high_size=40, scale=4)
Generator.load_weights(file_path)
print(Generator)

# data from ftp://cooler.csail.mit.edu/coolers/hg19/
name = os.path.join(path, raw_path, raw_file)
c = cooler.Cooler(name)
resolution = c.binsize
mat = c.matrix(balance=True).fetch('chr'+chromosome)

[Mh, _] = operations.remove_zeros(mat)
start = 0
end = 60
Mh = Mh[start:end, start:end]
print('MH: ', Mh.shape)

scale = 4
Ml = operations.sampling_hic(Mh, scale**2, fix_seed=True)
print('ML: ', Ml.shape)

# Normalization
# the input should not be type of np.matrix!
Ml = normalization.SCN_normalization(np.asarray(Ml), max_iter=3000)
Mh = normalization.SCN_normalization(np.asarray(Mh), max_iter=3000)

len_size = 40
hic_hr, index_1d_2d, index_2d_1d = operations.divide_pieces_hic(
    Mh, block_size=len_size, save_file=False)
hic_hr = np.asarray(hic_hr, dtype=np.float32)
print('shape hic_hr: ', hic_hr.shape)
hic_lr, _, _ = operations.divide_pieces_hic(
    Ml, block_size=len_size, save_file=False)
hic_lr = np.asarray(hic_lr, dtype=np.float32)
print('shape hic_lr: ', hic_lr.shape)
'''with np.load('./datasets_hic.npz', allow_pickle=True) as data:
    a = data['hic']
    b = data['index_1D_2D']
    c = data['index_2D_1D']
    h = merge_hic(a, b)'''

true_hic_hr = hic_hr
print('shape true hic_hr: ', true_hic_hr.shape)
[_, _, _, predict_hic_hr, _, _, _] = Generator(
    hic_lr[..., np.newaxis], training=False)
predict_hic_hr = np.squeeze(predict_hic_hr.numpy(), axis=3)
print(predict_hic_hr.shape)

sr_file += '_chr'+chromosome
file_path = os.path.join(directory_sr, sr_file)
np.savez_compressed(file_path+'.npz', predict_hic=predict_hic_hr,
                    true_hic=true_hic_hr, index_1D_2D=index_1d_2d, index_2D_1D=index_2d_1d, start_id=start, end_id=end)


predict_hic_hr_merge = operations.merge_hic(
    predict_hic_hr, index_1D_2D=index_1d_2d)
#predict_hic_hr_merge = normalization.SCN_normalization(predict_hic_hr_merge, max_iter=3000)
print('shape of merge predict hic hr', predict_hic_hr_merge.shape)

# chrop Mh
residual = Mh.shape[0] % int(len_size/2)
if residual > 0:
    Mh = Mh[0:-residual, 0:-residual]

diag = np.ones(Mh.shape) - np.diag(np.ones(Mh.shape[0])) - np.diag(
    np.ones(Mh.shape[0]-1), k=1) - np.diag(np.ones(Mh.shape[0]-1), k=-1)
Mh = Mh*diag
predict_hic_hr_merge = predict_hic_hr_merge*diag
print('sum Mh:', np.sum(np.abs(Mh)))
print('sum merge:', np.sum(np.abs(predict_hic_hr_merge)))
diff = np.abs(Mh-predict_hic_hr_merge)
print('sum diff: {:.5}'.format(np.sum(diff**2)))

fig, axs = plt.subplots(1, 2, figsize=(8, 15))
ax = axs[0].imshow(np.log1p(1000*predict_hic_hr_merge), cmap='RdBu_r')
axs[0].set_title('predict')
ax = axs[1].imshow(np.log1p(1000*Mh), cmap='RdBu_r')
axs[1].set_title('true')
plt.tight_layout()
plt.show()

predict_hic_coo = operations.dense2tag(predict_hic_hr_merge)
true_hic_coo = operations.dense2tag(Mh)
print('shape coo predict hic: {}'.format(predict_hic_coo.shape))
print('shape coo predict hic: {}'.format(true_hic_coo.shape))
