import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_hicgan
from software.utils import redircwd_back_projroot
from software.wrapper_hicgan import configure_hicgan, generate

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
input_path = 'pathto/refine_resolution/data/input_hicgan_2000000_40_28/'
preprocessing_output_path = ‘pathto/refine_resolution/data/input_hicgan_2000000_40_28/preprocessing_output/')
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = ‘pathto/refine_resolution/data/input_hicgan_2000000_40_28/train'
train_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
valid_path = ‘pathto/refine_resolution/data/input_hicgan_2000000_40_28/valid'
valid_list = ['17', '18', '22']
"""

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
 root_dir, experiment_name, preprocessing_chr_list, input_path,
 preprocessing_output_path, script_work_dir, train_path, train_list,
 valid_path, valid_list, predict_path, predict_list] = configure_hicgan()


prepare_hicgan.run(raw_hic=raw_hic,
                   chromosome_list=preprocessing_chr_list,
                   genomic_distance=genomic_distance,
                   lr_size=lr_size,
                   hr_size=hr_size,
                   downsample_factor=downsample_factor
                   )

chr_t_list = []
chr_v_list = []
lr_dir = os.path.join(preprocessing_output_path, 'lr')
hr_dir = os.path.join(preprocessing_output_path, 'hr')
for c in preprocessing_chr_list:
    if c in train_list:
        chr_t_list.append(c)
    if c in valid_list:
        chr_v_list.append(c)

generate(input_lr_dir=lr_dir, input_hr_dir=hr_dir,
         output_dir=train_path, postfix='train', chr_list=chr_t_list)
generate(input_lr_dir=lr_dir, input_hr_dir=hr_dir,
         output_dir=valid_path, postfix='valid', chr_list=chr_v_list)
generate(input_lr_dir=lr_dir, input_hr_dir=hr_dir,
         output_dir=predict_path, postfix='predict', chr_list=chr_p_list)
