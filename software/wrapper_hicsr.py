import os
import sys
import numpy as np
import subprocess
import shutil

from . import prepare_hicsr
from .utils import path_wrap, redircwd_back_projroot

"""test hicsr"""


def configure_hicsr():
    raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    genomic_distance = 2000000
    lr_size = 40
    hr_size = 28
    downsample_factor = 16

    methods_name = 'hicsr'
    root_dir = redircwd_back_projroot(project_name='refine_resolution')
    experiment_name = '_'.join(
        [methods_name, str(genomic_distance), str(lr_size), str(hr_size)])
    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    preprocessing_chr_list = ['22']
    data_cat = raw_hic.split('-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('.')[1]
    input_path = os.path.join(root_dir, 'data', 'input_'+experiment_name, data_cat)+'/'
    preprocessing_output_path = os.path.join(input_path, 'preprocessing_output/')

    script_work_dir = os.path.join(root_dir, 'software', 'HiCSR')

    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    train_path = os.path.join(input_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path, exist_ok=True)
    train_list = ['1', '2', '3', '4', '5', '6', '7', '8',
                  '9', '10', '11', '12', '13', '14', '15', '16', '22']
    valid_path = os.path.join(input_path, 'valid')
    if not os.path.exists(valid_path):
        os.makedirs(valid_path, exist_ok=True)
    valid_list = ['17', '18', '22']

    return raw_hic, genomic_distance, lr_size, hr_size, downsample_factor, \
        root_dir, experiment_name, preprocessing_chr_list, input_path, \
        preprocessing_output_path, script_work_dir, train_path, train_list, \
        valid_path, valid_list
