import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_hicsr
from .software.utils import path_wrap, redircwd_back_projroot
from .software.HiCSR import *

from wrapper_hicsr import configure_hicsr
"""test hicsr"""

"""raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
genomic_distance = 2000000
lr_size = 40
hr_size = 28
downsample_factor = 16

methods_name = 'hicsr'
root_dir = redircwd_back_projroot(project_name='refine_resolution')
experiment_name = '_'.join(
    [methods_name, str(genomic_distance), str(lr_size), str(hr_size)])
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
chr_list = ['22']"""

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
 root_dir, experiment_name, input_path, preprocessing_output_path,
 script_work_dir, train_path, train_list, valid_path, valid_list] = configure_hicsr()


prepare_hicsr.run(raw_hic=raw_hic,
                  chromosome_list=chr_list,
                  genomic_distance=genomic_distance,
                  lr_size=lr_size,
                  hr_size=hr_size,
                  downsample_factor=downsample_factor
                  )

# python preprocessing.py --input input_samples/ --output preprocessing_output/ --normalize 1
# input_samples/ --> input_hicsr_2000000_200/Rao2014_GM12878_10kb/
# preprocessing_output/ --> input_hicsr_2000000_200/Rao2014_GM12878_10kb/preprocessing_output/
# These sample matrices are stored in the input_samples directory, where each sample has the following naming convention
# <chromosome>-<cell_type>-<downsample_factor>-<file_tag>.txt.gz

"""input_path = os.path.join(root_dir, 'data', 'input_'+experiment_name)+'/'
preprocessing_output_path = os.path.join(
    root_dir, 'data', 'input_' + experiment_name, 'preprocessing_output/')"""

if os.path.exists(preprocessing_output_path):
    shutil.rmtree(preprocessing_output_path)
script = "preprocessing.py"
cmd = ["python", script, "--input", input_path,
       "--output", preprocessing_output_path, "--normalize", "1"]
print(' '.join(cmd))
process = subprocess.run(cmd, cwd=script_work_dir)

"""# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = os.path.join(root_dir, 'data', 'input_'+experiment_name, 'train')
if not os.path.exists(train_path):
    os.mkdir(train_path)
train_list = ['1', '2', '3', '4', '5', '6', '7', '8',
              '9', '10', '11', '12', '13', '14', '15', '16', '22']
valid_path = os.path.join(root_dir, 'data', 'input_'+experiment_name, 'valid')
if not os.path.exists(valid_path):
    os.mkdir(valid_path)
valid_list = ['17', '18', '22']"""

# move train and valid data to cp_path
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
