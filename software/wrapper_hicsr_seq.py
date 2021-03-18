import os
import sys
import numpy as np
import subprocess
import shutil

from . import prepare_hicsr
from .utils import redircwd_back_projroot, configure_seq

"""test hicsr"""


def configure_hicsr(raw_hic):
    # raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    # raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    genomic_distance = 2000000
    lr_size = 40
    hr_size = 28
    downsample_factor = 16

    methods_name = 'hicsr'

    [raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
     root_dir, experiment_name, input_path, script_work_dir] = configure_seq(
        raw_hic=raw_hic, genomic_distance=genomic_distance,
        lr_size=lr_size, hr_size=hr_size,
        downsample_factor=downsample_factor,
        methods_name=methods_name)

    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    if raw_hic in ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Shen2012-MouseCortex-HindIII-allreps-filtered.10kb.cool', 
        'Selvaraj2013-F123-HindIII-allreps-filtered.10kb.cool']:
        chr_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 'X']
    else:
        chr_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']

    preprocessing_chr_list = chr_list

    preprocessing_output_path = os.path.join(
        input_path, 'preprocessing_output/')

    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    train_path = os.path.join(input_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path, exist_ok=True)
    train_list = ['1', '2', '3', '4', '5', '6', '7', '8',
                  '9', '10', '11', '12', '13', '14', '15', '16']
    valid_path = os.path.join(input_path, 'valid')
    if not os.path.exists(valid_path):
        os.makedirs(valid_path, exist_ok=True)
    valid_list = ['17', '18']

    predict_path = os.path.join(input_path, 'predict')
    if not os.path.exists(predict_path):
        os.makedirs(predict_path, exist_ok=True)
    # predict_list = ['19', '20', '21', '22', 'X']
    predict_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']

    return raw_hic, genomic_distance, lr_size, hr_size, downsample_factor, \
        root_dir, experiment_name, preprocessing_chr_list, input_path, \
        preprocessing_output_path, script_work_dir, train_path, train_list, \
        valid_path, valid_list, predict_path, predict_list
