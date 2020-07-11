import os
import sys
import numpy as np
import subprocess
import shutil

from software import prepare_deephic
from software.utils import redircwd_back_projroot
from software.wrapper_deephic import configure_deephic

"""test deephic"""

"""e.g.
raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
genomic_distance = 2000000
lr_size = 40
hr_size = 28
downsample_factor = 16
methods_name = 'deephic'
root_dir = 'pathto/refine_resolution'
experiment_name = 'deephic_2000000_40_28'
chr_list = [‘1’, ..., '22']
input_path = 'pathto/refine_resolution/data/input_deephic_2000000_40_28/'
preprocessing_output_path = ‘pathto/refine_resolution/data/input_deephic_2000000_40_28/preprocessing_output/')
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
train_path = ‘pathto/refine_resolution/data/input_deephic_2000000_40_28/train'
train_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
valid_path = ‘pathto/refine_resolution/data/input_deephic_2000000_40_28/valid'
valid_list = ['17', '18', '22']
"""

[raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
 root_dir, experiment_name, chr_list, input_path, preprocessing_output_path,
 script_work_dir, train_path, train_list, valid_path, valid_list] = configure_deephic()


prepare_deephic.run(raw_hic=raw_hic,
                  chromosome_list=chr_list,
                  genomic_distance=genomic_distance,
                  lr_size=lr_size,
                  hr_size=hr_size,
                  downsample_factor=downsample_factor
                  )

"""if os.path.exists(preprocessing_output_path):
    shutil.rmtree(preprocessing_output_path)
script = "preprocessing.py"
cmd = ["python", script, "--input", input_path,
       "--output", preprocessing_output_path, "--normalize", "1"]
print(' '.join(cmd))
process = subprocess.run(cmd, cwd=script_work_dir)


# move train and valid data to cp_path
cp_path = os.path.join(preprocessing_output_path, 'deephic_dataset', 'samples')
for chro in chr_list:
    file = 'chr'+chro+'.npz'
    if chro in train_list:
        subprocess.run(["cp", os.path.join(cp_path, file),
                        os.path.join(train_path, file)])
    if chro in valid_list:
        subprocess.run(["cp", os.path.join(cp_path, file),
                        os.path.join(valid_path, file)])"""