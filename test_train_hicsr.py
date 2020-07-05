import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_hicsr
from software.utils import path_wrap, redircwd_back_projroot
from software.HiCSR import *
"""test hicsr"""

raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
genomic_distance = 2000000
lr_size = 40
hr_size = 28
downsample_factor = 16

methods_name = 'hicsr'
root_dir = redircwd_back_projroot(project_name='refine_resolution')
experiment_name = '_'.join([methods_name, str(genomic_distance), str(lr_size), str(hr_size)])
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
chr_list = ['22']

# python preprocessing.py --input input_samples/ --output preprocessing_output/ --normalize 1
# input_samples/ --> input_hicsr_2000000_200/Rao2014_GM12878_10kb/
# preprocessing_output/ --> input_hicsr_2000000_200/Rao2014_GM12878_10kb/preprocessing_output/
# These sample matrices are stored in the input_samples directory, where each sample has the following naming convention
# <chromosome>-<cell_type>-<downsample_factor>-<file_tag>.txt.gz

input_path = os.path.join(root_dir, 'data', 'input_'+experiment_name)+'/'
preprocessing_output_path = os.path.join(
    root_dir, 'data', 'input_' + experiment_name, 'preprocessing_output/')

if not os.path.exists(preprocessing_output_path):
    print("Preprocessing data First")

# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = os.path.join(root_dir, 'data', 'input_'+experiment_name, 'train')
if not os.path.exists(train_path):
    os.mkdir(train_path)
train_list = ['1', '2', '3', '4', '5', '6', '7', '8',
              '9', '10', '11', '12', '13', '14', '15', '16', '22']
valid_path = os.path.join(root_dir, 'data', 'input_'+experiment_name, 'valid')
if not os.path.exists(valid_path):
    os.mkdir(valid_path)
valid_list = ['17', '18', '22']

cp_path = os.path.join(root_dir, 'data', 'input_' + experiment_name,
                       'preprocessing_output', 'HiCSR_dataset', 'samples')
for chro in chr_list:
    file = 'chr'+chro+'-GM12878-HiCSR-dataset-normalized-samples.npz'
    if chro in train_list:
        subprocess.run(["cp", os.path.join(cp_path, file),
                        os.path.join(train_path, file)])
    if chro in valid_list:
        subprocess.run(["cp", os.path.join(cp_path, file),
                        os.path.join(valid_path, file)])

# training model
# python train.py --data_fp preprocessing_output/HiCSR_dataset/samples/ --model HiCSR --experiment test_HiCSR
data_fp = ""
model_hicsr = ""
cmd = ["python", "train.py", "--data_fp", data_fp, "--model",
       model_hicsr, "--experiment", "test_"+experiment_name]
process = subprocess.run(cmd, cwd=os.path.join(root_dir, 'software', 'HiCSR'))

# predict data
# python predict.py --input preprocessing_output/normalized/lr/ --output HiCSR_predictions/ --model_type HiCSR --model_fp pretrained_models/HiCSR.pth
