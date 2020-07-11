import os
import sys
import numpy as np
import subprocess
import shutil

import time
import multiprocessing


from . import prepare_deephic
from .utils import redircwd_back_projroot, configure

from software.DeepHiC.utils.io import compactM, divide, pooling
from software.DeepHiC import data_generate
"""test deephic"""


def configure_deephic():
    raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    genomic_distance = 2000000
    lr_size = 40
    hr_size = 40
    downsample_factor = 16

    methods_name = 'deephic'

    [raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
     root_dir, experiment_name, input_path, script_work_dir] = configure(
        raw_hic=raw_hic, genomic_distance=genomic_distance,
        lr_size=lr_size, hr_size=hr_size,
        downsample_factor=downsample_factor,
        methods_name=methods_name)

    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    preprocessing_chr_list = ['22']

    preprocessing_output_path = input_path

    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    train_path = os.path.join(input_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path, exist_ok=True)
    train_list = ['1', '2', '3', '4', '5', '6', '7', '8',
                  '9', '10', '11', '12', '13', '14', '15', '16', '22']
    valid_path = os.path.join(input_path, 'valid')
    if not os.path.exists(valid_path):
        os.makedirs(valid_path, exist_ok=True)
    valid_list = ['17', '18', '22']

    return raw_hic, genomic_distance, lr_size, hr_size, downsample_factor, \
        root_dir, experiment_name, preprocessing_chr_list, input_path, \
        preprocessing_output_path, script_work_dir, train_path, train_list, \
        valid_path, valid_list


# python data_generate.py -hr 10kb -lr 40kb -s all -chunk 40 -stride 40 -bound 201 -scale 1 -c GM12878
def generate(input_lr_dir, input_hr_dir, output_dir,
             cell_line='GM12878',
             high_res=10000,
             low_res=40000,
             chunk=40,
             stride=40,
             bound=201,
             scale=1,
             pool_type='max',
             chr_list=['22']):
    postfix = cell_line.lower()
    pool_str = 'nonpool' if scale == 1 else f'{pool_type}pool{scale}'
    print(
        f'Going to read {high_res} and {low_res} data, then deviding matrices with {pool_str}')

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    data_lr_dir = input_lr_dir
    data_hr_dir = input_hr_dir
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(
        f'Start a multiprocess pool with processes = {pool_num} for generating DeepHiC data')
    results = []
    for n in chr_list:
        high_file = os.path.join(data_hr_dir, f'chr{n}_{high_res}.npz')
        down_file = os.path.join(data_lr_dir, f'chr{n}_{low_res}.npz')
        kwargs = {'scale': scale, 'pool_type': pool_type, 'chunk': chunk,
                  'stride': stride, 'bound': bound, 'lr_cutoff': lr_cutoff}
        res = pool.apply_async(
            data_generate.deephic_divider, (n, high_file, down_file,), kwargs)
        results.append(res)
    pool.close()
    pool.join()
    print(
        f'All DeepHiC data generated. Running cost is {(time.time()-start)/60:.1f} min.')

    # return: n, div_dhic, div_hhic, div_inds, compact_idx, full_size
    data = np.concatenate([r.get()[1] for r in results])
    target = np.concatenate([r.get()[2] for r in results])
    inds = np.concatenate([r.get()[3] for r in results])
    compacts = {r.get()[0]: r.get()[4] for r in results}
    sizes = {r.get()[0]: r.get()[5] for r in results}

    filename = f'deephic_{high_res}{low_res}_c{chunk}_s{stride}_b{bound}_{pool_str}_{postfix}.npz'
    deephic_file = os.path.join(out_dir, filename)
    np.savez_compressed(deephic_file, data=data, target=target,
                        inds=inds, compacts=compacts, sizes=sizes)
    print('Saving file:', deephic_file)
