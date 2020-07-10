import os
import sys
import numpy as np
import subprocess
import shutil

from software.utils import redircwd_back_projroot

from software.wrapper_hicsr import configure_hicsr
"""test hicsr"""

"""e.g.
raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
genomic_distance = 2000000
lr_size = 40
hr_size = 28
downsample_factor = 16
methods_name = 'hicsr'
root_dir = 'pathto/refine_resolution'
experiment_name = 'hicsr_2000000_40_28'
chr_list = [‘1’, ..., '22']
input_path = 'pathto/refine_resolution/data/input_hicsr_2000000_40_28/'
preprocessing_output_path = ‘pathto/refine_resolution/data/input_hicsr_2000000_40_28/preprocessing_output/')
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = ‘pathto/refine_resolution/data/input_hicsr_2000000_40_28/train'
train_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
valid_path = ‘pathto/refine_resolution/data/input_hicsr_2000000_40_28/valid'
valid_list = ['17', '18', '22']
"""

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
 root_dir, experiment_name, chr_list, input_path, preprocessing_output_path,
 script_work_dir, train_path, train_list, valid_path, valid_list] = configure_hicsr()

# training model
# python train.py --data_fp preprocessing_output/HiCSR_dataset/samples/ --model HiCSR --experiment test_HiCSR
data_fp = ""
model_hicsr = ""
cmd = ["python", "train.py", "--data_fp", data_fp, "--model",
       model_hicsr, "--experiment", "test_"+experiment_name]
process = subprocess.run(cmd, cwd=script_work_dir)

# predict data
# python predict.py --input preprocessing_output/normalized/lr/ --output HiCSR_predictions/ --model_type HiCSR --model_fp pretrained_models/HiCSR.pth
