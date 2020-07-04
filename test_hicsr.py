import os
import sys
import numpy as np
import subprocess

from software import prepare_hicsr
from software.utils import path_wrap, redircwd_back_projroot
from software.HiCSR import *
"""test hicsr"""

raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
genomic_distance=2000000
lr_size=40
hr_size=28
downsample_factor=16

methods_name = 'hicsr'
root_dir = redircwd_back_projroot(project_name='refine_resolution')
experiment_name = '_'.join([methods_name, str(genomic_distance), str(lr_size), str(hr_size)])
chr_list =  ['22']#['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
prepare_hicsr.run(raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
                  chromosome_list=chr_list,
                  genomic_distance=2000000,
                  lr_size=40,
                  hr_size=28,
                  downsample_factor=16
                  )

# python preprocessing.py --input input_samples/ --output preprocessing_output/ --normalize 1
# input_samples/ --> input_hicsr_2000000_200/Rao2014_GM12878_10kb/
# preprocessing_output/ --> input_hicsr_2000000_200/Rao2014_GM12878_10kb/preprocessing_output/
# These sample matrices are stored in the input_samples directory, where each sample has the following naming convention
# <chromosome>-<cell_type>-<downsample_factor>-<file_tag>.txt.gz

input_path = os.path.join(root_dir, 'data', 'input_'+experiment_name)
input_path = path_wrap(input_path)
preprocessing_output_path = os.path.join(root_dir, 'data', 'input_' + experiment_name, 'preprocessing_output/')
preprocessing_output_path = path_wrap(preprocessing_output_path)
script = "preprocessing.py"
cmd = ["python ", script, "--input ", input_path,
       "--output ", preprocessing_output_path, "--normalize 1"]
print(' '.join(cmd))
process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, cwd=os.path.join(root_dir,'software', 'HiCSR'))
process.wait()

# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = ""
train_list = []
valid_path = ""
valid_list = []

# python train.py --data_fp preprocessing_output/HiCSR_dataset/samples/ --model HiCSR --experiment test_HiCSR

# python predict.py --input preprocessing_output/normalized/lr/ --output HiCSR_predictions/ --model_type HiCSR --model_fp pretrained_models/HiCSR.pth
