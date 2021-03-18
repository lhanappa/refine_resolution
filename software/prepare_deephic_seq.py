import os
import sys
import cooler
import numpy as np
from .utils import redircwd_back_projroot, cool_to_raw
from .utils import remove_zeros, sampling_hic


def save_to_compressed(hic, idx, output_path, output_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output = os.path.join(output_path, output_name)
    hic = np.asarray(hic)
    print(output)
    np.savez_compressed(output, hic=hic, compact=idx)



def run(raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        chromosome_list=['22'],
        genomic_distance=2000000,
        lr_size=40,
        hr_size=40,
        downsample_factor=16):
    methods_name = 'deephic'
    root_dir = redircwd_back_projroot(project_name='refine_resolution')
    experiment_name = '_'.join(
        [methods_name, str(genomic_distance), str(lr_size), str(hr_size)])
    data_cat = raw_hic.split('-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('-')[2] + '-' + str(downsample_factor) + '_' + raw_hic.split('.')[1]
    input_path = os.path.join(
        root_dir, 'data', 'input_'+experiment_name, data_cat)
    # preprocessing_output_path = os.path.join( root_dir, 'data', 'input_' + experiment_name, data_cat)

    cell_type = raw_hic.split('-')[1]

    [hic_m, hi_res] = cool_to_raw(raw_path=os.path.join(
        root_dir, 'data', 'raw'), raw_hic=raw_hic)

    low_res = int(np.sqrt(downsample_factor)*hi_res)

    for chro in chromosome_list:
        # chrN_10kb.npz
        name_hr = f'chr{chro}_{hi_res}.npz'
        chromosome = 'chr' + chro
        if chromosome not in hic_m.chromnames:
            continue
        mat_hr = hic_m.matrix(balance=True).fetch(chromosome)
        [mat_hr, idx] = remove_zeros(mat_hr)
        save_to_compressed(mat_hr, idx, output_path=os.path.join(
            input_path, 'hr'), output_name=name_hr)
        # chrN_40kb.npz
        name_lr = f'chr{chro}_{low_res}.npz'
        mat_lr = sampling_hic(mat_hr, downsample_factor, fix_seed=True)
        save_to_compressed(mat_lr, idx, output_path=os.path.join(
            input_path, 'lr'), output_name=name_lr)

if __name__ == '__main__':
    run()