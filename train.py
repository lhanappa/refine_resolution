import cooler
import numpy as np
import tensorflow as tf
from iced import normalization
import spektral

import matplotlib.pyplot as plt
from IPython import display
import time
tf.keras.backend.set_floatx('float32')

# data from ftp://cooler.csail.mit.edu/coolers/hg19/
name = 'Dixon2012-H1hESC-HindIII-allreps-filtered.100kb.cool'
#name = 'Rao2014-K562-MboI-allreps-filtered.500kb.cool'
c = cooler.Cooler(name)
resolution = c.binsize
mat= c.matrix(balance=True).fetch('chr2')
print(mat.shape)
idxy = ~np.all(np.isnan(mat),axis=0)
M = mat[idxy,:]
Mh = M[:,idxy]

IMG_HEIGHT, IMG_WIDTH = int(Mh.shape[0]/4),int(Mh.shape[1]/4)
img_l = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH))
for i in list(range(0, len(Mh))):
    x = int(np.floor(i/(len(Mh)/IMG_HEIGHT)))
    for j in list(range(0,len(Mh))):
        y = int(np.floor(j/(len(Mh)/IMG_WIDTH)))
        img_l[x, y] = img_l[x, y] + Mh[i,j]

Ml = img_l
plt.matshow(np.log2(Ml), cmap='YlOrRd')
print(Ml.shape)
import copy
Ml = img_l
#print('original Ml: ', Ml)
#Ml = normalization.ICE_normalization(Ml)
Ml = normalization.SCN_normalization(Ml)
Mh = normalization.SCN_normalization(Mh)
hic_lr = []
IMG_HEIGHT, IMG_WIDTH = int(512/4),int(512/4)
print(IMG_HEIGHT, IMG_WIDTH)
for i in range(len(Ml)-IMG_HEIGHT+1):
    hic_lr.append(Ml[i:i+IMG_HEIGHT, i:i+IMG_WIDTH])
hic_lr = np.array(hic_lr)
print(hic_lr.shape)
fig = plt.figure(figsize=(5, 5))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(np.log2(hic_lr[i*25,:,:]), cmap='YlOrRd')
    plt.axis('off')

hic_hr = []
IMG_HEIGHT, IMG_WIDTH = int(512),int(512)
print(IMG_HEIGHT, IMG_WIDTH)
for i in range(0,len(Mh)-IMG_HEIGHT+1,4):
    hic_hr.append(Mh[i:i+IMG_HEIGHT, i:i+IMG_WIDTH])
hic_hr = np.array(hic_hr)
print(hic_hr.shape)
fig = plt.figure(figsize=(5, 5))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(np.log2(hic_hr[i*25,:,:]), cmap='YlOrRd')
    plt.axis('off')

EPOCHS = 800
BUFFER_SIZE = 1
BATCH_SIZE = 3
train_data = tf.data.Dataset.from_tensor_slices((hic_lr[..., np.newaxis], hic_hr[..., np.newaxis])).batch(BATCH_SIZE)
train(train_data, EPOCHS, BATCH_SIZE)

import imageio
import glob
anim_file = 'gcn_1.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('./lvl2/image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)