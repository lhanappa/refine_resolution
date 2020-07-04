import os
import sys
import numpy as np
import subprocess

from software import prepare_hicsr

chr_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
prepare_hicsr.run(raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
                  chromosome_list=chr_list,
                  genomic_distance=2000000,
                  lr_size=40,
                  hr_size=28,
                  downsample_factor=16
                  )

# python preprocessing.py --input input_samples/ --output preprocessing_output/ --normalize 1
# input_samples/ --> input_hicsr_2000000_200/Rao2014_10kb/
# preprocessing_output/ --> input_hicsr_2000000_200/Rao2014_10kb/
# These sample matrices are stored in the input_samples directory, where each sample has the following naming convention
# <chromosome>-<cell_type>-<downsample_factor>-<file_tag>.txt.gz
# ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
cmd = ['python preprocessing.py' "--input input_samples/", "--output preprocessing_output/", "--normalize 1"]
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
process.wait()

# python train.py --data_fp preprocessing_output/HiCSR_dataset/samples/ --model HiCSR --experiment test_HiCSR

# python predict.py --input preprocessing_output/normalized/lr/ --output HiCSR_predictions/ --model_type HiCSR --model_fp pretrained_models/HiCSR.pth
