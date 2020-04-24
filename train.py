import model
import glob
import imageio
import time
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
print(mat.shape)
idxy = ~np.all(np.isnan(mat), axis=0)
M = mat[idxy, :]
Mh = M[:, idxy]

IMG_HEIGHT, IMG_WIDTH = int(Mh.shape[0]/4), int(Mh.shape[1]/4)
img_l = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH))
for i in list(range(0, len(Mh))):
    x = int(np.floor(i/(len(Mh)/IMG_HEIGHT)))
    for j in list(range(0, len(Mh))):
        y = int(np.floor(j/(len(Mh)/IMG_WIDTH)))
        img_l[x, y] = img_l[x, y] + Mh[i, j]

Ml = img_l
print('ML', Ml.shape)
Ml = img_l

# Normalization
Ml = normalization.SCN_normalization(Ml)
Mh = normalization.SCN_normalization(Mh)
hic_lr = []
IMG_HEIGHT, IMG_WIDTH = int(512/4), int(512/4)
print('Height: ', IMG_HEIGHT, 'Weight: ', IMG_WIDTH)
for i in range(len(Ml)-IMG_HEIGHT+1):
    hic_lr.append(Ml[i:i+IMG_HEIGHT, i:i+IMG_WIDTH])
hic_lr = np.array(hic_lr)
print('hic_lr: ', hic_lr.shape)

hic_hr = []
IMG_HEIGHT, IMG_WIDTH = int(512), int(512)
print('Height: ', IMG_HEIGHT, 'Weight: ', IMG_WIDTH)
for i in range(0, len(Mh)-IMG_HEIGHT+1, 4):
    hic_hr.append(Mh[i:i+IMG_HEIGHT, i:i+IMG_WIDTH])
hic_hr = np.array(hic_hr)
print('hic_hr: ', hic_hr.shape)


EPOCHS = 1000
BUFFER_SIZE = 1
BATCH_SIZE = 3
train_data = tf.data.Dataset.from_tensor_slices(
    (hic_lr[..., np.newaxis], hic_hr[..., np.newaxis])).batch(BATCH_SIZE)

Gen = model.make_generator_model()
Dis = model.make_discriminator_model()
print(Gen.summary())
tf.keras.utils.plot_model(Gen, to_file='G.png', show_shapes=True)
print(Dis.summary())
tf.keras.utils.plot_model(Dis, to_file='D.png', show_shapes=True)

model.train(Gen, Dis, train_data, EPOCHS, BATCH_SIZE)

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
