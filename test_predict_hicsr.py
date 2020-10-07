import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_hicsr
#from software.utils import redircwd_back_projroot
from software.HiCSR import *

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
input_path = 'pathto/refine_resolution/data/input_hicsr_2000000_40_28/Rao2014_GM12878_10kb/'
preprocessing_output_path = ‘pathto/refine_resolution/data/input_hicsr_2000000_40_28/Rao2014_GM12878_10kb/preprocessing_output/')
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = ‘pathto/refine_resolution/data/input_hicsr_2000000_40_28/Rao2014_GM12878_10kb/train'
train_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
valid_path = ‘pathto/refine_resolution/data/input_hicsr_2000000_40_28/Rao2014_GM12878_10kb/valid'
valid_list = ['17', '18', '22']
"""

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
 root_dir, experiment_name, chr_list, input_path, preprocessing_output_path,
 script_work_dir, train_path, train_list, valid_path, valid_list, predict_path, predict_list] = configure_hicsr()

# predict data
# python predict.py --input preprocessing_output/normalized/lr/ --output HiCSR_predictions/ --model_type HiCSR --model_fp pretrained_models/HiCSR.pth
data_fp = predict_path
model_fp = os.path.join(input_path, 'model', 'HiCSR', 'HiCSR.pth')
model_hicsr = "HiCSR"

data_cat = raw_hic.split('-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('.')[1]
output_path = os.path.join(root_dir, 'data', 'output_'+experiment_name, data_cat, 'SR')+'/'

cmd = ["python", "predict.py", 
        "--input", data_fp, 
        "--output", output_path, 
        "--model_type", "HiCSR", 
        "--model_fp", model_fp]
process = subprocess.run(cmd, cwd=script_work_dir)