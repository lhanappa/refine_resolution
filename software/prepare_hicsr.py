import os
import sys
import cooler
from .utils import redircwd_back_projroot, cool_to_raw
from .utils import remove_zeros, sampling_hic

import numpy as np


def save_to_raw(hic, output_path, output_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output = os.path.join(output_path, output_name)
    hic = np.asarray(hic)
    print(output)
    np.savetxt(output, hic)



def run(raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        chromosome_list=['22'],
        genomic_distance=2000000,
        lr_size=40,
        hr_size=28,
        downsample_factor=16
        ):

    methods_name = 'hicsr'
    root_dir = redircwd_back_projroot(project_name='refine_resolution')
    experiment_name = '_'.join(
        [methods_name, str(genomic_distance), str(lr_size), str(hr_size)])
    data_cat = raw_hic.split(
        '-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('.')[1]
    input_path = os.path.join(
        root_dir, 'data', 'input_'+experiment_name, data_cat)
    # preprocessing_output_path = os.path.join( root_dir, 'data', 'input_' + experiment_name, data_cat)

    cell_type = raw_hic.split('-')[1]

    file_tag = 'sample'
    [hic_m, _] = cool_to_raw(raw_path=os.path.join(
        root_dir, 'data', 'raw'), raw_hic=raw_hic)

    if not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)

    # <chromosome>-<cell_type>-<downsample_factor>-<file_tag>.txt.gz
    for chro in chromosome_list:
        name_hr = '-'.join(['chr'+chro, cell_type,
                            str(1), file_tag]) + '.txt.gz'
        chromosome = 'chr' + chro
        mat_hr = hic_m.matrix(balance=True).fetch(chromosome)
        [mat_hr, _] = remove_zeros(mat_hr)
        save_to_raw(mat_hr, output_path=os.path.join(
            input_path, 'hr'), output_name=name_hr)

        name_lr = '-'.join(['chr'+chro, cell_type,
                            str(downsample_factor), file_tag]) + '.txt.gz'
        mat_lr = sampling_hic(mat_hr, downsample_factor, fix_seed=True)
        save_to_raw(mat_lr, output_path=os.path.join(
            input_path, 'lr'), output_name=name_lr)

    # python preprocessing.py --input input_samples/ --output preprocessing_output/ --normalize 1
    # input_samples/ --> input_hicsr_2000000_200/Rao2014_10kb/
    # preprocessing_output/ --> input_hicsr_2000000_200/Rao2014_10kb/
    # These sample matrices are stored in the input_samples directory, where each sample has the following naming convention
    # <chromosome>-<cell_type>-<downsample_factor>-<file_tag>.txt.gz
    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

    # python train.py --data_fp preprocessing_output/HiCSR_dataset/samples/ --model HiCSR --experiment test_HiCSR

    # python predict.py --input preprocessing_output/normalized/lr/ --output HiCSR_predictions/ --model_type HiCSR --model_fp pretrained_models/HiCSR.pth


if __name__ == '__main__':
    run()
