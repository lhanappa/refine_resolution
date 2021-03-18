import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_hicsr
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

'''raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
        'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool']'''

# 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool',6
raw_list = ['Shen2012-MouseCortex-HindIII-allreps-filtered.10kb.cool', 
        'Selvaraj2013-F123-HindIII-allreps-filtered.10kb.cool']
idx = int(sys.argv[1])

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
 root_dir, experiment_name, chr_list, input_path, preprocessing_output_path,
 script_work_dir, train_path, train_list, valid_path, valid_list, predict_path, predict_list] = configure_hicsr(raw_list[idx])


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

if os.path.exists(preprocessing_output_path):
    shutil.rmtree(preprocessing_output_path)
script = "preprocessing.py"
cmd = ["python", script, "--input", input_path,
       "--output", preprocessing_output_path, "--normalize", "1", "--input_size", str(lr_size), "--output_size", str(hr_size)]
print(' '.join(cmd))
process = subprocess.run(cmd, cwd=script_work_dir)

cell_type = raw_hic.split('-')[1]
# move train and valid data to cp_path
cp_path = os.path.join(preprocessing_output_path, 'HiCSR_dataset', 'samples')
for chro in chr_list:
    file = 'chr{}-{}-HiCSR-dataset-normalized-samples.npz'.format(chro, cell_type)
    if chro in train_list:
        subprocess.run(["cp", os.path.join(cp_path, file),
                        os.path.join(train_path, file)])
    if chro in valid_list:
        subprocess.run(["cp", os.path.join(cp_path, file),
                        os.path.join(valid_path, file)])

# move predict 'txt' file to cp_path
cp_path = os.path.join(preprocessing_output_path, 'normalized','lr')
for chro in chr_list:
    file = 'chr{}-{}-16-norm.txt.gz'.format(chro, cell_type)
    if chro in predict_list:
        subprocess.run(["cp", os.path.join(cp_path, file),
                        os.path.join(predict_path, file)])
