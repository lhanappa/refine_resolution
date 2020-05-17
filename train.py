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

import tensorflow as tf
tf.keras.backend.set_floatx('float32')


# data from ftp://cooler.csail.mit.edu/coolers/hg19/
name = 'Dixon2012-H1hESC-HindIII-allreps-filtered.100kb.cool'
#name = 'Rao2014-K562-MboI-allreps-filtered.500kb.cool'
c = cooler.Cooler(name)
resolution = c.binsize
mat = c.matrix(balance=True).fetch('chr2')
idxy = ~np.all(np.isnan(mat), axis=0)
M = mat[idxy, :]
Mh = M[:, idxy]
print('MH: ', Mh.shape)
scale = 4
len_size = 128
IMG_HEIGHT, IMG_WIDTH = int(Mh.shape[0]/scale), int(Mh.shape[1]/scale)
img_l = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH))
for i in list(range(0, len(Mh))):
    x = int(np.floor(i/(len(Mh)/IMG_HEIGHT)))
    for j in list(range(0, len(Mh))):
        y = int(np.floor(j/(len(Mh)/IMG_WIDTH)))
        img_l[x, y] = img_l[x, y] + Mh[i, j]

Ml = img_l
print('ML: ', Ml.shape)

# Normalization
Ml = normalization.SCN_normalization(Ml)
Mh = normalization.SCN_normalization(Mh)

hic_lr = []
IMG_HEIGHT, IMG_WIDTH = int(len_size/scale), int(len_size/scale)
print('Height: ', IMG_HEIGHT, 'Weight: ', IMG_WIDTH)
Ml_h, Ml_w = Ml.shape
block_height = int(IMG_HEIGHT/2);
block_width = int(IMG_WIDTH/2);
Ml_d0 = np.split(Ml, np.arange(block_height, Ml_h, block_height), axis=0)
Ml_d1 = list(map(lambda x: np.split(x, np.arange(block_width, Ml_w, block_width), axis=1), Ml_d0))
hic_half_l = np.array(Ml_d1)
hic_half_l = hic_half_l[0:-1,0:-1]
print('hic_lr: ', hic_half_l.shape)
for dis in np.arange(1, hic_half_l.shape[0]):
    for i in np.arange(0, hic_half_l.shape[1]-dis):
        hic_lr.append(np.block([[hic_half_l[i,i], hic_half_l[i,i+dis]],[hic_half_l[i+dis,i], hic_half_l[i+dis,i+dis]]]))
print('len hic_lr: ', len(hic_lr))

hic_hr = []
IMG_HEIGHT, IMG_WIDTH = int(len_size), int(len_size)
print('Height: ', IMG_HEIGHT, 'Weight: ', IMG_WIDTH)
Mh_h, Mh_w = Mh.shape
block_height = int(IMG_HEIGHT/2);
block_width = int(IMG_WIDTH/2);
Mh_d0 = np.split(Mh, np.arange(block_height, Mh_h, block_height), axis=0)
Mh_d1 = list(map(lambda x: np.split(x, np.arange(block_width, Mh_w, block_width), axis=1), Mh_d0))
hic_half_h = np.array(Mh_d1)
hic_half_h = hic_half_h[0:-1,0:-1]
print('hic_hr: ', hic_half_h.shape)
for dis in np.arange(1, hic_half_h.shape[0]):
    for i in np.arange(0, hic_half_h.shape[1]-dis):
        hic_hr.append(np.block([[hic_half_h[i,i], hic_half_h[i,i+dis]],[hic_half_h[i+dis,i], hic_half_h[i+dis,i+dis]]]))
print('len hic_hr: ', len(hic_hr))

'''plt.figure(figsize=(5,8))
plt.subplot(2,3,1)
plt.imshow(np.log2(hic_lr[0]))
plt.subplot(2,3,2)
plt.imshow(np.log2(hic_lr[1]))
plt.subplot(2,3,3)
plt.imshow(np.log2(hic_lr[2]))
plt.subplot(2,3,4)
plt.imshow(np.log2(hic_hr[0]))
plt.subplot(2,3,5)
plt.imshow(np.log2(hic_hr[1]))
plt.subplot(2,3,6)
plt.imshow(np.log2(hic_hr[2]))
plt.show()'''

EPOCHS = 3000
BUFFER_SIZE = 1
BATCH_SIZE = 9
len_low_size = 32
scale = 4
hic_lr = np.array(hic_lr)
hic_hr = np.array(hic_hr)
train_data = tf.data.Dataset.from_tensor_slices((hic_lr[0::2,..., np.newaxis], hic_hr[0::2,..., np.newaxis])).batch(BATCH_SIZE)
test_data = tf.data.Dataset.from_tensor_slices((hic_lr[1::2,..., np.newaxis], hic_hr[1::2,..., np.newaxis])).batch(BATCH_SIZE)

Gen = model.make_generator_model(len_low_size=len_low_size, scale=scale)
Dis = model.make_discriminator_model(len_low_size=len_low_size, scale=scale)
print(Gen.summary())
tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
print(Dis.summary())
tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#Gen.save('./saved_model/'+current_time+'/gen_model') 
#Dis.save('./saved_model/'+current_time+'dis_model')
model.train(Gen, Dis, train_data, EPOCHS, len_low_size, scale, test_data)

Gen.save('./saved_model/'+current_time+'/gen_model') 
Dis.save('./saved_model/'+current_time+'dis_model')

'''anim_file = 'gan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('./lvl2/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)'''
