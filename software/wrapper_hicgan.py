import os
import sys
import numpy as np
import subprocess
import shutil

import time
import multiprocessing


from . import prepare_hicgan
from .utils import redircwd_back_projroot, configure

from software.hicGAN import train_hicgan
"""test hicgan"""


def configure_hicgan():
    raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    genomic_distance = 2000000
    lr_size = 40
    hr_size = 40
    downsample_factor = 16

    methods_name = 'hicgan'

    [raw_hic, genomic_distance, lr_size, hr_size, downsample_factor,
     root_dir, experiment_name, input_path, script_work_dir] = configure(
        raw_hic=raw_hic, genomic_distance=genomic_distance,
        lr_size=lr_size, hr_size=hr_size,
        downsample_factor=downsample_factor,
        methods_name=methods_name)

    # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    preprocessing_chr_list = ['21', '22']

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
             postfix='train',
             high_res=10000,
             low_res=40000,
             chunk=40,
             stride=40,
             bound=201,
             chr_list=['22']):
    postfix = postfix.lower()

    pool_num = 23 if multiprocessing.cpu_count() > 23 else multiprocessing.cpu_count()

    data_lr_dir = input_lr_dir
    data_hr_dir = input_hr_dir
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    start = time.time()
    pool = multiprocessing.Pool(processes=pool_num)
    print(
        f'Start a multiprocess pool with processes = {pool_num} for generating hicgan data')
    results = []
    for n in chr_list:
        high_file = os.path.join(data_hr_dir, f'chr{n}_{high_res}.npz')
        down_file = os.path.join(data_lr_dir, f'chr{n}_{low_res}.npz')
        kwargs = {'chunk': chunk, 'stride': stride, 'bound': bound}
        if n.isnumeric():
            chrn = int(n)
        else:
            chrn = n
        res = pool.apply_async(
            prepare_hicgan.hicgan_divider, (chrn, high_file, down_file,), kwargs)
        results.append(res)
    pool.close()
    pool.join()
    print(
        f'All hicgan data generated. Running cost is {(time.time()-start)/60:.1f} min.')

    # return: n, div_dhic, div_hhic, div_inds, compact_idx, full_size
    data = np.concatenate([r.get()[1] for r in results])
    data = np.transpose(data, axes=(0,2,3,1))
    target = np.concatenate([r.get()[2] for r in results])
    target = np.transpose(target, axes=(0,2,3,1))
    filename = f'hicgan_{high_res}{low_res}_c{chunk}_s{stride}_b{bound}_{postfix}.npz'
    hicgan_file = os.path.join(out_dir, filename)
    np.savez_compressed(hicgan_file, lr_data=data, hr_data=target)
    print('Saving file:', hicgan_file)


def train(train_dir, valid_dir, model_dir, lr, hr, chunk, stride, bound, num_epochs, cwd_dir=None):
    if cwd_dir is not None:
        os.chdir(cwd_dir)
    print("cwd: ", os.getcwd())
    print("train_dir: ", train_dir)
    print("valid_dir: ", valid_dir)
    print("model_dir: ", model_dir)
    print("train hicgan start")
    resos = str(hr)+str(lr)
    train_dir = os.path.join(train_dir, f'hicgan_{resos}_c{chunk}_s{stride}_b{bound}_train.npz')
    valid_dir = os.path.join(valid_dir, f'hicgan_{resos}_c{chunk}_s{stride}_b{bound}_valid.npz')
    train_hicgan.train(train_dir, valid_dir, model_dir, num_epochs=num_epochs)
