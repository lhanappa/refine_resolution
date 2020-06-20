import time
import datetime
import cooler
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from iced import normalization
from utils import operations, quality_hic

path = './data'
raw_path = 'raw'
raw_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
#'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
chromosome = '22'
max_distance = 2000000
resolution = 10000

sr_path = 'output'
sr_file = raw_file.split('-')[0] + '_' + raw_file.split('.')[1]
directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)

input_path = directory_sr
input_file = sr_file+'_chr'+chromosome+'.npz'
if not os.path.exists(os.path.join(input_path, input_file)):
    print('not input file')

print('input path: ', input_path, input_file)
with np.load(os.path.join(input_path, input_file), allow_pickle=True) as data:
    predict_hic = data['predict_hic']
    true_hic = data['true_hic']
    idx_1d_2d = data['index_1D_2D'][()]  # get dict()
    idx_2d_1d = data['index_2D_1D'][()]  # get dict()
    start = data['start_id']
    end = data['end_id']

print(end+1)
print(idx_1d_2d)
predict_hic_hr_merge = operations.merge_hic(predict_hic, index_1D_2D=idx_1d_2d)
print('shape of merge predict hic hr', predict_hic_hr_merge.shape)

true_hic_hr_merge = operations.merge_hic(true_hic, index_1D_2D=idx_1d_2d)
print('shape of merge predict hic hr', predict_hic_hr_merge.shape)

k = np.ceil(max_distance/resolution).astype(int)

true_hic_hr_merge = operations.filter_diag_boundary(true_hic_hr_merge, diag_k=2, boundary_k=k)
predict_hic_hr_merge = operations.filter_diag_boundary(predict_hic_hr_merge, diag_k=2, boundary_k=k)
print('sum true:', np.sum(np.abs(true_hic_hr_merge)))
print('sum predict:', np.sum(np.abs(predict_hic_hr_merge)))
diff = np.abs(true_hic_hr_merge-predict_hic_hr_merge)
print('sum diff: {:.5}'.format(np.sum(diff**2)))

fig, axs = plt.subplots(1, 2, figsize=(8, 15))
ax = axs[0].imshow(np.log1p(1000*predict_hic_hr_merge), cmap='RdBu_r')
axs[0].set_title('predict')
ax = axs[1].imshow(np.log1p(1000*true_hic_hr_merge), cmap='RdBu_r')
axs[1].set_title('true')
plt.tight_layout()
plt.show()

operations.format_bin(true_hic_hr_merge, coordinate=(
    0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/demo.bed.gz')
operations.format_contact(true_hic_hr_merge, coordinate=(
    0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/demo_contact_true.gz')
operations.format_contact(predict_hic_hr_merge, coordinate=(
    0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/demo_contact_predict.gz')

#quality_hic.configure_file(input_path+'/', 'demo', 'p1', 'p2', input_path+'/demo_contact_true.gz', input_path+'/demo_contact_predict.gz')