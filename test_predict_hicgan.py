import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_hicgan
from software.utils import redircwd_back_projroot
from software.wrapper_hicgan import configure_hicgan, generate, predict

"""test hicgan"""

"""e.g.
raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
genomic_distance = 2000000
lr_size = 40
hr_size = 28
downsample_factor = 16
methods_name = 'hicgan'
root_dir = 'pathto/refine_resolution'
experiment_name = 'hicgan_2000000_40_28'
chr_list = [‘1’, ..., '22']
input_path = 'pathto/refine_resolution/data/input_hicgan_2000000_40_40/'
preprocessing_output_path = ‘pathto/refine_resolution/data/input_hicgan_2000000_40_40/preprocessing_output/')
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = ‘pathto/refine_resolution/data/input_hicgan_2000000_40_40/train'
train_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
valid_path = ‘pathto/refine_resolution/data/input_hicgan_2000000_40_40/valid'
valid_list = ['17', '18']
predict_path = ‘pathto/refine_resolution/data/input_hicgan_2000000_40_40/predict'
preditc_list = ['19', '20', '21', '22', 'X']
"""

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
    root_dir, experiment_name, preprocessing_chr_list, input_path,
    preprocessing_output_path, script_work_dir, train_path, train_list,
    valid_path, valid_list, predict_path, predict_list] = configure_hicgan()

"""chr_list = []
lr_dir = os.path.join(preprocessing_output_path, 'lr')
hr_dir = os.path.join(preprocessing_output_path, 'hr')
for c in preprocessing_chr_list:
    if c in train_list:
        chr_list.append(c)
generate(input_lr_dir=lr_dir, input_hr_dir=hr_dir, output_dir=train_path, chr_list=chr_list)"""

data_cat = raw_hic.split('-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('.')[1]
output_path = os.path.join(root_dir, 'data', 'output_'+experiment_name, data_cat, 'SR')+'/'

saved_cpt_dir = os.path.join(preprocessing_output_path, 'model', 'ckpt')
if not os.path.exists(saved_cpt_dir):
    print('Can\'t find model! Please check directory')
files = [f for f in os.listdir(saved_cpt_dir)]
cpt_name = [f for f in files if f.find('g_hicgan_GM12878_weights.npz') >= 0][0]
predict(data_dir=predict_path, out_dir=output_path,
        ckpt_file=os.path.join(saved_cpt_dir, cpt_name),
        lr=40000, 
        cwd_dir=script_work_dir)
