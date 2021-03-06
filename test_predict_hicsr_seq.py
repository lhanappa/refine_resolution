import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_hicsr
# from software.HiCSR import *
# from software.utils import redircwd_back_projroot
from software.wrapper_hicsr_seq import configure_hicsr
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
'''raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
        'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool']'''

'''raw_list = ['Shen2012-MouseCortex-HindIII-allreps-filtered.10kb.cool', 
        'Selvaraj2013-F123-HindIII-allreps-filtered.10kb.cool']'''

raw_list = ['Rao2014-GM12878-MboI-allreps-filtered.10kb.cool']
idx = 0

downsample_factor = int(sys.argv[1])

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
 root_dir, experiment_name, chr_list, input_path, preprocessing_output_path,
 script_work_dir, train_path, train_list, valid_path, valid_list, predict_path, predict_list] = configure_hicsr(raw_list[idx], downsample_factor)

print('train path: ', train_path)
print('valid path: ', valid_path)
print('predict path: ', predict_path)

# predict data
# python predict.py --input preprocessing_output/normalized/lr/ --output HiCSR_predictions/ --model_type HiCSR --model_fp pretrained_models/HiCSR.pth
data_fp = predict_path
# model_fp = os.path.join(input_path, 'model', 'HiCSR', 'HiCSR.pth')
model_fp = os.path.join(root_dir, 'trained_model', 'hicsr', 'model', 'HiCSR', 'HiCSR.pth')
model_hicsr = "HiCSR"

data_cat = raw_hic.split('-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('-')[2] + '-' + str(downsample_factor) + '_' + raw_hic.split('.')[1]
output_path = os.path.join(root_dir, 'data', 'output_'+experiment_name, data_cat, 'SR')+'/'
resolution = 10000
cmd = ["python", "predict_hicsr.py", 
        "--input", data_fp, 
        "--output", output_path, 
        "--model_type", "HiCSR", 
        "--model_fp", model_fp,
        "--resolution", str(resolution)]
process = subprocess.run(cmd, cwd=script_work_dir)